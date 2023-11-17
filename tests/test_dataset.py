''' simple test of the images_mapping func
'''
import unittest
import numpy as np
from brain_age_prediction.useful_functions import load_dataset


class Tests(unittest.TestCase):
    '''
    Unittest for loading datasets
    '''

    def test_structural_dataset(self):
        '''
        check shape of structural dataset
        '''

        df_s_td, df_s_asd = load_dataset(dataset_name='Harmonized_structural_features.csv')

        shape_s_td = df_s_td.shape
        shape_s_asd = df_s_asd.shape

        np.testing.assert_equal(shape_s_td,(703,226))
        np.testing.assert_equal(shape_s_asd,(680,226))


    def test_functional_dataset(self):
        '''
        check shape of functional dataset
        '''

        df_f_td, df_f_asd = load_dataset(dataset_name='Harmonized_functional_features.csv')

        shape_f_td = df_f_td.shape
        shape_f_asd = df_f_asd.shape
        print(shape_f_asd)
        np.testing.assert_array_equal(shape_f_td,(703,5258))
        np.testing.assert_array_equal(shape_f_asd,(680,5258))

if __name__ == '__main__':
    unittest.main()
