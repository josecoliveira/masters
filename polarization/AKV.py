import numpy as np


class AKV:
    def check_sizes(self):
        if self.belief_array.shape[0] != self.influence_graph.shape[0]:
            raise Exception(
                "Number of agents in belief_state and in the influence graph are different."
            )

        if self.influence_graph.shape[0] != self.influence_graph[1]:
            raise Exception("Influcence graph must be a square matrix")

    def check_values(self):
        if np.any(np.sum(self.belief_array, axis=1) != 0):
            raise Exception("All belief states must sum up to 1.")

        if np.any(self.influence_graph < 0 or self.influence_graph > 1):
            raise Exception("Influence must be between 0 or 1")

        if np.any(self.belief_array < 0 or self.belief_array > 1):
            raise Exception("Influence must be between 0 or 1")

    def __init__(self, belief_array, influence_graph):
        self.belief_array = np.array(belief_array)
        self.influence_graph = np.array(influence_graph)

        self.check_sizes()

        self.a = self.belief_array[0]
        self.k = self.belief_array[1]

        self.check_values()

        self.states = np.array([belief_array])

    @property
    def number_of_agents(self):
        return self.a

    @property
    def domain_size(self):
        return self.k

    @staticmethod
    def classic_update(belief_state_i, belief_state_j, influence):
        return belief_state_i + influence * (belief_state_j - belief_state_i)

    def overall_class_update(self):
        new_belief_array = np.array(
            [
                [
                    1
                    / self.a
                    * np.sum(
                        [
                            AKV.classic_update(
                                self.belief_array[ai, ki],
                                self.belief_array[aj, ki],
                                self.influence_graph[aj, ai],
                            )
                            for aj in range(self.a)
                        ]
                    )
                    for ki in range(self.k)
                ]
                for ai in range(self.a)
            ]
        )
        self.belief_array = new_belief_array
        self.states = np.vstack((self.states, self.belief_array))
        return self.belief_array

    def get_polarization(self, k=201, K=1000, alpha=1.0):
        pis = np.array(
            [np.histogram(self.belief_array[:, i], bins=k) for i in range(self.k)]
        )

        def polarization(pi):
            return K * np.sum(
                [
                    np.sum([pi[i] ** (1 - alpha) * pi[j] for j in range(k)])
                    for i in range(k)
                ]
            )

        return np.array([polarization(pis[i]) for i in range(self.k)])
