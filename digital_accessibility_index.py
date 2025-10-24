import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class DigitalAccessibilityIndex:
    """A class to compute the Digital Accessibility Index (DAI) for villages."""
    
    def __init__(self, data, parameters, weights=None):
        """
        Initialize the DAI calculator.
        
        Args:
            data (pd.DataFrame): DataFrame with villages as rows and parameters as columns.
            parameters (list): List of column names representing DAI parameters.
            weights (dict, optional): Predefined weights for parameters. If None, PCA is used.
        """
        self.data = data[parameters].copy()
        self.parameters = parameters
        self.weights = weights
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.imputer = KNNImputer(n_neighbors=5)
        self.index_scores = None
        
    def clean_data(self):
        """
        Handle missing data using KNN imputation and flag unreliable data.
        
        Returns:
            pd.DataFrame: Cleaned and imputed data.
        """
        # Flag rows with missing data
        self.data['missing_data_flag'] = self.data.isna().any(axis=1).astype(int)
        
        # Perform KNN imputation
        imputed_data = self.imputer.fit_transform(self.data[self.parameters])
        self.data[self.parameters] = imputed_data
        
        return self.data
    
    def normalize_data(self):
        """
        Normalize the data using StandardScaler.
        
        Returns:
            pd.DataFrame: Normalized data.
        """
        self.data[self.parameters] = self.scaler.fit_transform(self.data[self.parameters])
        return self.data
    
    def calculate_weights_pca(self):
        """
        Calculate weights using PCA based on normalized data.
        
        Returns:
            dict: Normalized weights for each parameter.
        """
        # Apply PCA
        self.pca.fit(self.data[self.parameters])
        
        # Calculate weights: sum of (eigenvectors * explained variance ratio)
        loadings = self.pca.components_
        explained_variance = self.pca.explained_variance_ratio_
        weights = np.sum(loadings.T * explained_variance, axis=1)
        
        # Normalize weights to sum to 1
        weights_normalized = weights / np.sum(weights)
        
        # Store weights as dictionary
        self.weights = dict(zip(self.parameters, weights_normalized))
        return self.weights
    
    def compute_dai(self):
        """
        Compute the Digital Accessibility Index for each village.
        
        Returns:
            pd.Series: DAI scores for each village.
        """
        if self.weights is None:
            self.calculate_weights_pca()
        
        # Calculate DAI as weighted sum of normalized parameters
        self.index_scores = np.zeros(len(self.data))
        for param, weight in self.weights.items():
            self.index_scores += weight * self.data[param]
        
        # Store DAI scores in DataFrame
        self.data['DAI'] = self.index_scores
        
        return self.data['DAI']
    
    def validate_weights(self, expert_weights=None):
        """
        Validate PCA-derived weights against expert weights or perform sensitivity analysis.
        
        Args:
            expert_weights (dict, optional): Expert-provided weights for comparison.
        
        Returns:
            dict: Comparison of PCA weights with expert weights (if provided) and sensitivity metrics.
        """
        validation_results = {'pca_weights': self.weights}
        
        if expert_weights:
            # Compare PCA weights with expert weights
            weight_diff = {param: self.weights.get(param, 0) - expert_weights.get(param, 0) 
                          for param in self.parameters}
            validation_results['weight_differences'] = weight_diff
            
        # Sensitivity analysis: perturb weights and check index variation
        sensitivity = []
        original_dai = self.index_scores.copy()
        
        for param in self.parameters:
            temp_weights = self.weights.copy()
            temp_weights[param] *= 1.1  # Increase weight by 10%
            temp_weights = {p: w / sum(temp_weights.values()) for p, w in temp_weights.items()}
            
            # Recalculate DAI with perturbed weights
            temp_dai = np.zeros(len(self.data))
            for p, w in temp_weights.items():
                temp_dai += w * self.data[p]
            
            # Calculate mean absolute change in DAI
            sensitivity.append(np.mean(np.abs(temp_dai - original_dai)))
        
        validation_results['sensitivity'] = dict(zip(self.parameters, sensitivity))
        return validation_results
    
    def get_results(self):
        """
        Return the final DataFrame with DAI scores and flags.
        
        Returns:
            pd.DataFrame: DataFrame containing original data, DAI scores, and flags.
        """
        return self.data

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'Village': ['Village1', 'Village2', 'Village3'],
        'Internet_Access': [0.8, 0.6, 0.4],
        'Mobile_Penetration': [0.7, 0.9, 0.5],
        'Internet_Banking': [0.5, 0.4, 0.3]
    })
    
    parameters = ['Internet_Access', 'Mobile_Penetration', 'Internet_Banking']
    
    # Initialize DAI calculator
    dai = DigitalAccessibilityIndex(data, parameters)
    
    # Clean and normalize data
    dai.clean_data()
    dai.normalize_data()
    
    # Calculate weights and DAI
    weights = dai.calculate_weights_pca()
    dai_scores = dai.compute_dai()
    
    # Validate weights
    expert_weights = {'Internet_Access': 0.40, 'Mobile_Penetration': 0.30, 'Internet_Banking': 0.20}
    validation = dai.validate_weights(expert_weights)
    
    # Get results
    results = dai.get_results()
    
    print("Weights:", weights)
    print("DAI Scores:\n", dai_scores)
    print("Validation Results:", validation)
    print("Final Results:\n", results)