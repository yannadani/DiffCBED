import numpy as np

# import strategies.multi_perturbation_ed.main as mpe_main
import strategies.multi_perturbation_ed.finite_cd as finite_cd

from .acquisition_strategy import AcquisitionStrategy


class SSFinite(AcquisitionStrategy):
    def acquire(self, nodes, iteration):
        print("Values are set to be fixed")
        values = (
            np.zeros((self.args.batch_size, self.args.num_nodes))
            + self.args.intervention_value
        )

        # nodes = np.random.choice([True, False], size=self.args.batch_size*self.args.num_nodes).reshape(self.args.batch_size, self.args.num_nodes)

        # put dags in a format that it's friendly for the algorithm

        nnodes = self.args.num_nodes

        # K: number of samples per intervention you give
        # M:samples per simulation in the objective
        M = 20
        K = 20

        b = self.args.batch_size
        k = self.args.num_targets

        bs_dags = []
        for i in range(len(self.model.dags)):
            _dag = dict()
            est_dag = self.model.dags[i].to_amat()
            # A, b = finite.get_weights_MLE(est_dag, obs_samples)
            # bs_dags.append({'dag': est_dag, 'A': A, 'b': b, 'w': (1/num_bs), 'count':1})

            _dag["dag"] = (est_dag > 0).astype(np.int)
            _dag["A"] = est_dag
            _dag["b"] = self.model.dags[i].variances
            _dag["w"] = np.exp(self.model.normalized_log_weights[i])
            bs_dags.append(_dag)

        ss_loss = finite_cd.MI_obj_gen(int(M / 10), bs_dags, np.zeros(nnodes) + 5, K=K)

        interventions = finite_cd.ss_finite(bs_dags, ss_loss, b, k, True)

        nodes = np.zeros((self.args.batch_size, self.args.num_nodes))

        for i, intervention in enumerate(interventions):
            for node in intervention:
                nodes[i][node] = True

        # ss_loss = MI_obj_gen(int(M/10), bs_dags,np.zeros(nnodes)+5, K=K)
        #                 inter=ss_finite(bs_dags, ss_loss, b, k, True)

        # finite_cd.ss_infinite(bs_dags, "MI", self.args.batch_size, self.args.num_targets, True, all_k=True)

        return {"nodes": nodes, "values": values}, None
