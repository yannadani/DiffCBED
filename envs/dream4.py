import glob
import json
import os
import pathlib
import shutil
import subprocess
import uuid
from collections import namedtuple
from pathlib import Path
from xml.dom import minidom

import graphical_models
import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from .causal_environment import CausalEnvironment, Data
from .utils import expm_np, num_mec


def create_tmp(base):
    uid = str(uuid.uuid4())
    tmp_path = Path(base) / uid
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def observe(samples, path, name="insilico_size10_1"):
    network = f"../configurations/{name}.xml"
    cmd = f"java -jar ../gnw-3.1b.jar -c ../settings.txt --input-net {network} --output-net-format=1 --simulate"

    data = []
    os.makedirs("envs/dream4/tmp", exist_ok=True)
    for _ in tqdm.tqdm(range(samples)):
        tmp_path = create_tmp("envs/dream4")
        subprocess.check_call(cmd.split(" "), cwd=tmp_path, stderr=subprocess.DEVNULL)
        data.append(pd.read_csv(f"{tmp_path}/{name}_wildtype.tsv", sep="\t"))

        shutil.rmtree(tmp_path)
    return pd.concat(data).to_numpy()


def intervene(node, path, name="insilico_size10_1"):
    os.makedirs("envs/dream4/tmp", exist_ok=True)
    network = f"../configurations/{name}.xml"
    cmd = f"java -jar ../gnw-3.1b.jar -c ../settings.txt --input-net {network} --output-net-format=1 --simulate"
    tmp_path = create_tmp("envs/dream4")
    subprocess.check_call(cmd.split(" "), cwd=tmp_path, stderr=subprocess.DEVNULL)
    data = pd.read_csv(f"{tmp_path}/{name}_knockouts.tsv", sep="\t")
    shutil.rmtree(tmp_path)
    return data.iloc[node].to_numpy()


def get_network(xml):
    xmldoc = minidom.parse(str(xml))

    nodes = []
    var2id = {}
    for i, node in enumerate(xmldoc.getElementsByTagName("species")):
        name = node.attributes.get("id").value
        if "void" not in name:
            nodes.append(name)
            var2id[name] = i

    A = np.zeros((len(nodes), len(nodes)))

    for node in xmldoc.getElementsByTagName("reaction"):
        # child
        child = (
            node.getElementsByTagName("listOfProducts")[0]
            .getElementsByTagName("speciesReference")[0]
            .attributes.get("species")
            .value
        )

        # parents
        for parent in node.getElementsByTagName("modifierSpeciesReference"):
            _from = var2id[parent.attributes.get("species").value]
            _to = var2id[child]
            A[_from, _to] = 1

    return nodes, var2id, A


class OnlyDAGDream4Environment(CausalEnvironment):
    def __init__(
        self,
        args,
        noise_type="isotropic-gaussian",
        noise_sigma=1.0,
        node_range=[-10, 10],
        num_samples=1000,
        mu_prior=2.0,
        sigma_prior=1.0,
        seed=10,
        path="envs/dream4/configurations",
        name="insilico_size10_1",
        nonlinear=False,
        binary_nodes=False,
        logger=None,
    ):
        self.noise_sigma = noise_sigma
        self.logger = logger

        self.path = path
        self.name = name
        self.nodes, self.var2id, A = get_network(pathlib.Path(path) / f"{name}.xml")
        self.adjacency_matrix = A
        self.num_nodes = len(self.nodes)

        self.id2var = {}
        for key, val in self.var2id.items():
            self.id2var[val] = key

        self.graph = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.DiGraph)
        self.labeled_graph = nx.relabel_nodes(self.graph, self.id2var)
        self.dag = graphical_models.DAG.from_nx(self.graph)

        self.xml_file = pathlib.Path(path) / f"{name}.xml"
        self.held_out_data = np.array([])
        self.nodes = self.dag.nodes
        self.arcs = self.dag.arcs

        super().__init__(
            args,
            self.num_nodes,
            len(self.graph.edges),
            noise_type,
            num_samples,
            node_range=node_range,
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
            seed=seed,
            nonlinear=nonlinear,
            binary_nodes=binary_nodes,
            logger=logger,
        )

        self.reseed(self.seed)
        self.init_sampler()

    def __getitem__(self, index):
        return self.samples[index]

    def dag(self):
        return graphical_models.DAG.from_nx(self.graph)
