import numpy as np
import cvxpy as cp
from toqito.random import random_density_matrix, random_state_vector, random_unitary
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NotCloseError(Exception):
    pass


def random_centroids(N, C):
    # N: Dimension
    # C: Number of states

    sigma_list = []
    for i in range(C):
        sigma = random_density_matrix(N)
        sigma_list.append(sigma)
    return sigma_list


def distribution(C):
    distribution = np.random.rand(C)
    distribution = distribution / sum(distribution)
    return distribution


def POVM(sigma_list, probs):  # PGM
    N = len(sigma_list[0])
    C = len(probs)
    S = np.sum([probs[i] * sigma_list[i] for i in range(C)], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues_inv_sqrt = np.diag([1.0 / np.sqrt(l.real) if np.abs(l) > 1e-10 else 0 for l in eigenvalues])
    S_inv_sqrt = eigenvectors @ eigenvalues_inv_sqrt @ eigenvectors.conj().T

    E_list = []
    for i in range(C):
        E_i = S_inv_sqrt @ (probs[i] * sigma_list[i]) @ S_inv_sqrt
        E_list.append(E_i)

    return E_list


def POVM_NC_innerDistance_purity(N, C):
    sigma_list = random_centroids(N, C)
    probs = distribution(C)
    E_list = POVM(sigma_list, probs)
    E_sum = np.sum(E_list, axis=0)
    if not np.allclose(E_sum, np.identity(N)):
        raise NotCloseError("E_i do not sum to identity!")
    success_prob = 0
    for i in range(len(E_list)):
        # success_prob += np.trace(probs[i] * E_list[i] @ sigma_list[i])
        success_prob += (probs[i] * np.sum(E_list[i] * sigma_list[i].T))
    if not np.allclose(success_prob, success_prob.real):
        raise NotCloseError("Success probability is not real!")

    return sigma_list, probs, inner_distance_list(sigma_list), success_prob.real, purity_list(sigma_list)


def inner_distance_list(sigma_list):
    C = len(sigma_list)
    inn_distance_list = []
    for i in range(C - 1):
        for j in range(i + 1, C):
            inn_distance_list.append(np.linalg.norm(sigma_list[i] - sigma_list[j], 'fro') ** 2)
    return inn_distance_list


def purity_list(sigma_list):  # new HS distance (not cosine similarity)
    C = len(sigma_list)
    _purity_list = []
    for i in range(C):
        _purity_list.append(np.linalg.norm(sigma_list[i], 'fro') ** 2)
    return _purity_list


def better(sigma_list, probs, weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = len(sigma_list[0])
    C = len(probs)
    S = sum([weights[i] * sigma_list[i] for i in range(C)])
    eigenvalues, eigenvectors = torch.linalg.eigh(S)
    eigenvalues = eigenvalues.to(dtype=torch.complex128)
    eigenvectors = eigenvectors.to(dtype=torch.complex128).to(device)
    eigenvalues_inv_sqrt = torch.diag(torch.tensor([1.0 / torch.sqrt(l.real) if torch.abs(l.real) > 1e-10 else 0 for l in eigenvalues]))
    eigenvalues_inv_sqrt = eigenvalues_inv_sqrt.to(dtype=torch.complex128).to(device)
    S_inv_sqrt = eigenvectors @ eigenvalues_inv_sqrt @ eigenvectors.conj().transpose(-2, -1)

    E_list = []
    for i in range(C):
        E_i = S_inv_sqrt @ (weights[i] * sigma_list[i]) @ S_inv_sqrt
        E_i = E_i.to(dtype=torch.complex128)
        E_list.append(E_i)

    E_sum = sum(E_list)
    if not torch.allclose(E_sum, torch.eye(N, dtype=torch.complex128).to(device)):
        raise NotCloseError("E_i do not sum to identity!")
    success_prob = torch.tensor(0, dtype=torch.complex128).to(device)
    for i in range(len(E_list)):
        success_prob += (probs[i].to(dtype=torch.complex128) * torch.sum(E_list[i].to(dtype=torch.complex128) * (sigma_list[i].T).to(dtype=torch.complex128)))
    if not torch.allclose(success_prob, success_prob.real.to(dtype=torch.complex128)):
        print(success_prob)
        raise NotCloseError("Success probability is not real!")
    return success_prob.real


def SDP(sigma_list, probs):
    N = len(sigma_list[0])
    C = len(probs)

    # SDP variables
    E = [cp.Variable((N, N), hermitian=True) for _ in range(C)]

    # Objective function
    objective = cp.Maximize(cp.real(cp.sum([probs[i] * cp.trace(sigma_list[i] @ E[i]) for i in range(C)])))

    # Constraints
    constraints = [E[i] >> 0 for i in range(C)]  # E[i] must be positive semidefinite
    constraints.append(sum(E) == np.eye(N))  # The POVM elements must sum up to the identity

    # Define and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    return [problem.value] + [E[i].value for i in range(C)]


class FidProbData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
