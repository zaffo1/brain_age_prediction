'''
Test the shape of the imported dataset.
Important note:
since the used data are confidential, only a limited sample was uploaded to GitHub.
Therefore, to pass the test in the GitHub workflow, the test has to be made on the shape
of the sample dataset, which is, of couse, different from the shape of the complete dataset.
Before changing the dimensions to match the sample ones, the 'real' test was run locally.
'''
import unittest
import numpy as np
from brain_age_prediction.utils.loading_data import load_dataset

class Tests(unittest.TestCase):
    '''
    Unittest for loading datasets
    '''

    def test_structural_dataset(self):
        '''
        Check the shape of the structural dataset.

        This test ensures that the loaded structural dataset has the expected shape.
        It compares the shape of the dataset for both the TD and ASD groups.

        Sample dataset (expected shape):
        - TD: (0, 226)
        - ASD: (10, 226)

        Complete dataset (expected shape):
        - TD: (703, 226)
        - ASD: (680, 226)
        '''

        df_s_td, df_s_asd = load_dataset(dataset_name='sample_Harmonized_structural_features.csv')
        #df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

        shape_s_td = df_s_td.shape
        shape_s_asd = df_s_asd.shape
        #sample dataset
        np.testing.assert_equal(shape_s_td,(0,226))
        np.testing.assert_equal(shape_s_asd,(10,226))
        #complete dataset:
        #np.testing.assert_equal(shape_s_td,(703,226))
        #np.testing.assert_equal(shape_s_asd,(680,226))

    def test_functional_dataset(self):
        '''
        Check the shape of the functional dataset.

        This test ensures that the loaded functional dataset has the expected shape.
        It compares the shape of the dataset for both the TD and ASD groups.

        Sample dataset (expected shape):
        - TD: (0, 5258)
        - ASD: (10, 5258)

        Complete dataset (expected shape):
        - TD: (703, 5258)
        - ASD: (680, 5258)
        '''
        df_f_td, df_f_asd = load_dataset(dataset_name='sample_Harmonized_functional_features.csv')
        #df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

        shape_f_td = df_f_td.shape
        shape_f_asd = df_f_asd.shape
        #sample dataset
        np.testing.assert_array_equal(shape_f_td,(0,5258))
        np.testing.assert_array_equal(shape_f_asd,(10,5258))
        #complete dataset
        #np.testing.assert_array_equal(shape_f_td,(703,5258))
        #np.testing.assert_array_equal(shape_f_asd,(680,5258))

    def test_targets(self):
        '''
        Check if the targets of the structural and functional dataframes are the same.

        This test ensures that the age targets in the structural and functional dataframes
        for both the TD and ASD groups are identical.

        Sample dataset (age targets):
        - TD (Structural): (empty array, no samples)
        - ASD (Structural): (10 samples)
        - TD (Functional): (empty array, no samples)
        - ASD (Functional): (10 samples)

        Complete dataset (age targets):
        - TD (Structural): (703 samples)
        - ASD (Structural): (680 samples)
        - TD (Functional): (703 samples)
        - ASD (Functional): (680 samples)
        '''
        df_s_td, df_s_asd = load_dataset(dataset_name='sample_Harmonized_structural_features.csv')
        #df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

        df_f_td, df_f_asd = load_dataset(dataset_name='sample_Harmonized_functional_features.csv')
        #df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

        y_s_td = np.array(df_s_td['AGE_AT_SCAN'])
        y_f_td = np.array(df_f_td['AGE_AT_SCAN'])

        y_s_asd = np.array(df_s_asd['AGE_AT_SCAN'])
        y_f_asd = np.array(df_f_asd['AGE_AT_SCAN'])

        np.testing.assert_array_equal(y_s_td,y_f_td)
        np.testing.assert_array_equal(y_s_asd,y_f_asd)

if __name__ == '__main__':
    unittest.main()
