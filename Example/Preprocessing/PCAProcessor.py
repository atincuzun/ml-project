class PCAProcessor:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def apply_pca(self, landmarks):
        data = np.array(landmarks)
        mean = np.mean(data, axis=0)
        centered_data = data - mean
        covariance_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_components = eigenvectors[:, sorted_indices[:self.n_components]]
        transformed_data = centered_data @ principal_components
        return transformed_data