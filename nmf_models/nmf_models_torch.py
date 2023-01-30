import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

class NMF(nn.Module):

    def __init__(self, n_topics, lr=1e-2, betas=[0.5, 0.6], lamda=0.1,
                 epochs=100):
        """
        In the constructor we instantiate W and H, using N (n. rows in Y),
        D (n. cols in Y) and K (latent topics).
        """
        super().__init__()
        self.k = n_topics
        self.lr = lr
        self.betas = betas
        self.lamda = lamda
        self.epochs = epochs
        self.loss = []

    def forward(self):
        """
        reconstruct X from theta and phi
        """

        sp_theta = self.my_softplus1(self.theta)
        sp_phi = self.my_softplus1(self.phi)
        return ((torch.matmul(sp_theta, sp_phi))
                / (torch.matmul(sp_theta, sp_phi) + 1))

    def loss_func(self, target, predict):
        loss = torch.norm((target - predict), p='fro')
        # theta_F_norm = torch.norm(phi, p='fro')
        # phi_F_norm = torch.norm(phi, p='fro')
        # theta_col_norm  = torch.sum(theta, 0).mean()   #columns
        # phi_row_norm  = torch.sum(phi, 1).mean()  #rows
        return loss  # + 0.001*theta_col_norm + 0.001*phi_row_norm

    def fit(self, atac_mat):
        self.cells = atac_mat.shape[0]
        self.regions = atac_mat.shape[1]

        self.theta = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                                 high=1.0,
                                                                 size=(self.cells, self.k))
                                               ).type(torch.FloatTensor),
                                  requires_grad=True)  # cell-topic

        self.phi = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                               high=1.0,
                                                               size=(self.k, self.regions))
                                             ).type(torch.FloatTensor),
                                requires_grad=True)  # topic-feature

        self.optimizer = torch.optim.Adam([self.theta, self.phi], lr=self.lr,
                                          betas=self.betas)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0
        for i in range(self.epochs):
            if counter == interval:
                counter = 0
                progress += 1
                print('{}/10 through. Current error is {}'.format(progress, cost))

            pred_regions = self.forward()

            cost = self.loss_func(atac_mat, pred_regions)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            self.loss.append(cost.detach())
            try:
                if self.loss[-2] < self.loss[-1]:
                    early_stopper += 1
                elif early_stopper > 0:
                    early_stopper = 0

                if early_stopper > 200:
                    break
            except IndexError:
                continue

            counter += 1

        del pred_regions
        del atac_mat

    def my_softplus1(self, x):
        return torch.log(1.0 + torch.exp(x))


class intNMF(nn.Module):

    def __init__(self, n_topics, lr=1e-2, betas=[0.8, 0.8], lamda=0.1,
                 epochs=100):
        """
        In the constructor we instantiate W and H, using N (n. rows in rna and
        atac mat), D (n. cols in Y) and K (latent topics). Rows should
        correspond to cells and columns to features
        """
        super().__init__()

        self.k = n_topics
        self.lr = lr
        self.betas = betas
        self.lamda = lamda
        self.epochs = epochs
        self.loss = []

    def forward(self):
        """
        reconstruct X from theta and phi. Assume both TF-IDF transformed
        (positive numbers)
        """
        pred_regions = (torch.matmul((self.my_softplus(self.v_atac)
                        * self.my_softplus(self.theta))
                        , self.my_softplus(self.phi_atac)))
        pred_genes = (torch.matmul((self.my_softplus(self.v_rna)
                      * self.my_softplus(self.theta))
                      , self.my_softplus(self.phi_rna)))
        return pred_regions, pred_genes

    def loss_func(self, rna_mat, atac_mat, pred_genes, pred_regions):
        return (torch.norm((rna_mat - pred_genes), p='fro')
                + torch.norm((atac_mat - pred_regions), p='fro'))
        # + self.lambda* (regularise)

    def fit(self, rna_mat, atac_mat):
        cells = atac_mat.shape[0]
        regions = atac_mat.shape[1]
        genes = rna_mat.shape[1]

        RNA_mat = torch.tensor(rna_mat).type(torch.FloatTensor)
        ATAC_mat = torch.tensor(atac_mat).type(torch.FloatTensor)

        self.theta = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                                 high=1.0,
                                                                 size=(cells, self.k))
                                               ).type(torch.FloatTensor),
                                  requires_grad=True)  # cell-topic

        self.phi_atac = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                                    high=1.0,
                                                                    size=(self.k, regions))
                                                  ).type(torch.FloatTensor),
                                     requires_grad=True)  # feature-topic

        self.phi_rna = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                                   high=1.0,
                                                                   size=(self.k, genes))
                                                 ).type(torch.FloatTensor)
                                    , requires_grad=True)  # feature-topic

        self.v_atac = nn.Parameter(torch.tensor(np.ones((cells, 1)))
                                   .type(torch.FloatTensor),
                                   requires_grad=True)
        self.v_rna = nn.Parameter(torch.tensor(np.ones((cells, 1)))
                                  .type(torch.FloatTensor),
                                  requires_grad=True)

        self.optimizer = torch.optim.Adam([self.theta, self.phi_atac,
                                           self.phi_rna, self.v_atac,
                                           self.v_rna],
                                          lr=self.lr, betas=self.betas)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0
        for i in range(self.epochs):

            if counter == interval:
                counter = 0
                progress += 1
                print('{}/10 through. Current error is {}'.format(progress, cost))

            pred_regions, pred_genes = self.forward()

            cost = self.loss_func(RNA_mat, ATAC_mat, pred_genes, pred_regions)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            self.loss.append(cost.detach())
            try:
                if self.loss[-2] < self.loss[-1]:
                    early_stopper += 1
                elif early_stopper > 0:
                    early_stopper = 0

                if early_stopper > 200:
                    break
            except IndexError:
                continue

            counter += 1

        del pred_regions
        del pred_genes
        del RNA_mat
        del ATAC_mat

    def my_softplus(self, x):
        return torch.log(1.0 + torch.exp(x))


class intNMF_kmeans(nn.Module):

    def __init__(self, n_topics, lr=1e-2, betas=[0.5, 0.6], lamda=0.1,
                 epochs=100):
        """
        In the constructor we instantiate W and H, using N (n. rows in rna
        and atac mat), D (n. cols in Y) and K (latent topics). Rows should
        correspond to cells and columns to features
        """

        super().__init__()

        self.k = n_topics

        self.lr = lr
        self.betas = betas
        self.lamda = lamda
        self.epochs = epochs
        self.loss = []

    def forward(self):
        """
        reconstruct X from theta and phi. Assume both TF-IDF transformed
        (positive numbers)
        """

        pred_regions = (torch.matmul((self.my_softplus(self.v_atac)
                        * self.my_softplus(self.theta)),
                        self.my_softplus(self.phi_atac)))
        pred_genes = (torch.matmul((self.my_softplus(self.v_rna)
                      * self.my_softplus(self.theta)),
                      self.my_softplus(self.phi_rna)))
        return pred_genes, printNMF_alternatinged_regions

    def loss_func(self, rna_mat, atac_mat, pred_genes, pred_regions):
        return (torch.norm((rna_mat - pred_genes), p='fro')
                + torch.norm((atac_mat - pred_regions), p='fro'))
        # + self.lambda* (regularise)

    def fit(self, rna_mat, atac_mat):
        cells = atac_mat.shape[0]
        regions = atac_mat.shape[1]
        genes = rna_mat.shape[1]

        init_atac = KMeans(n_clusters=self.k, random_state=0).fit(atac_mat)
        init_rna = KMeans(n_clusters=self.k, random_state=0).fit(rna_mat)

        self.theta = nn.Parameter(torch.tensor(np.random.uniform(low=0.0,
                                                                 high=1.0,
                                                                 size=(cells, self.k))
                                               ).type(torch.FloatTensor)
                                  , requires_grad=True)  # cell-topic

        self.phi_atac = nn.Parameter(torch.tensor(init_atac.cluster_centers_)
                                     .type(torch.FloatTensor),
                                     requires_grad=True)  # feature-topic
        self.phi_rna = nn.Parameter(torch.tensor(init_rna.cluster_centers_)
                                    .type(torch.FloatTensor),
                                    requires_grad=True)  # feature-topic

        self.v_atac = nn.Parameter(torch.tensor(np.ones((cells, 1)))
                                   .type(torch.FloatTensor),
                                   requires_grad=True)
        self.v_rna = nn.Parameter(torch.tensor(np.ones((cells, 1)))
                                  .type(torch.FloatTensor),
                                  requires_grad=True)

        RNA_mat = torch.tensor(rna_mat).type(torch.FloatTensor)
        ATAC_mat = torch.tensor(atac_mat).type(torch.FloatTensor)

        self.optimizer = torch.optim.Adam([self.theta, self.phi_atac,
                                           self.phi_rna, self.v_atac,
                                           self.v_rna],
                                          lr=self.lr, betas=self.betas)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0
        for i in range(self.epochs):
            if counter == interval:
                counter = 0
                progress += 1
                print('{}/10 through. Current error is {}'.format(progress, cost))

            pred_genes, pred_regions = self.forward()

            cost = self.loss_func(RNA_mat, ATAC_mat, pred_genes, pred_regions)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            self.loss.append(cost.detach())
            try:
                if self.loss[-2] < self.loss[-1]:
                    early_stopper += 1
                elif early_stopper > 0:
                    early_stopper = 0

                if early_stopper > 20:
                    break
            except IndexError:
                continue

            counter += 1

        del pred_regions
        del pred_genes
        del RNA_mat
        del ATAC_mat

    def my_softplus(self, x):
        return torch.log(1.0 + torch.exp(x))


class intNMF_alternating(nn.Module):

    def __init__(self, n_topics, lr=1e-2, betas=[0.5, 0.6], lamda=0.1,
                 epochs=100):
        """
        In the constructor we instantiate W and H, using N (n. rows in rna and
        atac mat), D (n. cols in Y) and K (latent topics). Rows should
        correspond to cells and columns to features.
        """
        super().__init__()

        self.k = n_topics

        self.lr = lr
        self.betas = betas
        self.lamda = lamda
        self.epochs = epochs
        self.loss = []

    def forward(self):
        """
        reconstruct X from theta and phi. Assume both TF-IDF transformed
        (positive numbers)
        """
        pred_regions = (torch.matmul((self.my_softplus(self.v_atac)
                        * self.my_softplus(self.theta)),
                        self.my_softplus(self.phi_atac)))
        pred_genes = (torch.matmul((self.my_softplus(self.v_rna)
                      * self.my_softplus(self.theta)),
                      self.my_softplus(self.phi_rna)))
        return pred_genes, pred_regions

    def loss_func(self, rna_mat, atac_mat, pred_genes, pred_regions):
        return (torch.norm((rna_mat - pred_genes), p='fro')
                + torch.norm((atac_mat - pred_regions), p='fro'))
        # + self.lambda* (regularise)

    def fit(self, rna_mat, atac_mat):
        self.cells = atac_mat.shape[0]
        self.regions = atac_mat.shape[1]
        self.genes = rna_mat.shape[1]

        init_atac = KMeans(n_clusters=self.k, random_state=0).fit(atac_mat)
        init_rna = KMeans(n_clusters=self.k, random_state=0).fit(rna_mat)

        self.theta = nn.Parameter(torch.tensor(
                                   np.random.uniform(low=0.0,
                                                     high=1.0,
                                                     size=(self.cells, self.k))
                                               ).type(torch.FloatTensor),
                                  requires_grad=True)  # cell-topic

        self.phi_atac = nn.Parameter(torch.tensor(init_atac.cluster_centers_)
                                     .type(torch.FloatTensor),
                                     requires_grad=True)  # feature-topic

        self.phi_rna = nn.Parameter(torch.tensor(init_rna.cluster_centers_)
                                    .type(torch.FloatTensor),
                                    requires_grad=True)  # feature-topic

        self.v_atac = nn.Parameter(torch.tensor(np.ones((self.cells, 1)))
                                   .type(torch.FloatTensor),
                                   requires_grad=True)

        self.v_rna = nn.Parameter(torch.tensor(np.ones((self.cells, 1)))
                                  .type(torch.FloatTensor),
                                  requires_grad=True)

        RNA_mat = torch.tensor(rna_mat).type(torch.FloatTensor)
        ATAC_mat = torch.tensor(atac_mat).type(torch.FloatTensor)

        self.optimizer_theta = torch.optim.Adam([self.theta, self.v_atac,
                                                 self.v_rna],
                                                lr=self.lr, betas=self.betas)
        self.optimizer_phi = torch.optim.Adam([self.phi_atac, self.phi_rna],
                                              lr=self.lr, betas=self.betas)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0
        for i in range(self.epochs):
            if counter == interval:
                counter = 0
                progress += 1
                print('{}/10 through. Current error is {}'.format(progress, cost))

            pred_genes, pred_regions = self.forward()

            cost = self.loss_func(RNA_mat, ATAC_mat, pred_genes, pred_regions)


            #        """Calculate the updates. Alternate optimising theta and phi every 10 steps
            self.optimizer_theta.zero_grad()
            self.optimizer_phi.zero_grad()
            cost.backward()
            if ((i // 10)%2) == 0:
                self.optimizer_theta.step()
            else:
                self.optimizer_phi.step()

            self.loss.append(cost.detach())
            try:
                if self.loss[-2] < self.loss[-1]:
                    early_stopper += 1
                elif early_stopper > 0:
                    early_stopper = 0

                if early_stopper > 20:
                    break
            except IndexError:
                continue

            counter += 1

        del pred_regions
        del pred_genes
        del RNA_mat
        del ATAC_mat


    def my_softplus(self, x):
        return torch.log(1.0 + torch.exp(x))

if __name__ == 'main':
    print('main')
