import cngi.dio
import cngi.vis
import unittest
import xarray as xr
import numpy as np

class DdiJoinBase(unittest.TestCase):

    def setUp(self):
        try:
            self.vis_xds0 = cngi.dio.read_vis("../data/sis14_twhya_calibrated_flagged.vis.zarr", ddi=0)
            self.global_xds = cngi.dio.read_vis("../data/sis14_twhya_calibrated_flagged.vis.zarr", ddi='global')
        except Exception as ee:
            raise RuntimeError("These tests assume that sis14_twhya_calibrated_flagged.ms has been previously converted and stored in the ../data directory.") from ee
        # limit data a bit to speed up test times
        self.vis_xds0 = self.vis_xds0.where(self.vis_xds0.time >= self.vis_xds0.time[-10]).dropna("time")
        self.vis_xds0 = self.vis_xds0.where(self.vis_xds0.baseline < 10).dropna("baseline")

    def helper_get_joinable_ddis(self, cppy_change_times, deep_copy_both=False):
        """I don't have an obvious pair of compatible DDIs to join, so I create
        one compatible DDI by starting with a copy of an existing one."""

        # create the ddi copy
        orig = self.vis_xds0
        if deep_copy_both:
            orig = self.vis_xds0.copy(deep=True)
        cppy = orig.copy(deep=True)

        # start the copy's times 1 second after the end of the original's end times
        if cppy_change_times:
            start_time = orig.time[-1] + 10**9 # time stored as nanoseconds, so start 1 second after the last time
            delta_time = start_time - orig.time[0]
            cppy = cppy.assign_coords(time=orig.time + delta_time)

        return orig, cppy

    def helper_add_nondim_coord(self, xds, coord_name, parent_dim_name=None):
        if parent_dim_name == None:
            parent_dim_name = list(xds.dims.keys())[0]
        parent_dim_len = xds.dims[parent_dim_name]
        ret = xds.assign_coords({coord_name: xr.DataArray(range(parent_dim_len), dims=parent_dim_name)})
        self.assertTrue(coord_name in ret.coords, f"Dataset should have the \"{coord_name}\" coordinate!")
        self.assertFalse(coord_name in ret.dims, f"The coordinate \"{coord_name}\" should not be a dimension coordinate!")
        return ret, parent_dim_len

    def helper_add_dim_coord(self, xds, coord_name, length=None, values=None):
        if length != None:
            values = range(length)
        ret = xds.assign_coords({coord_name: values})
        self.assertTrue(coord_name in ret.coords, f"Dataset should have the \"{coord_name}\" coordinate!")
        self.assertTrue(coord_name in ret.dims, f"The coordinate \"{coord_name}\" should be a dimension coordinate!")
        return ret, len(values)

    def helper_add_data_var(self, xds, data_var_name, dim_name):
        dim_len = len(xds.coords[dim_name])
        ret = xds.assign({data_var_name: xr.DataArray(range(dim_len), dims=dim_name)})
        return ret, dim_len

    def helper_get_nonnan_index(self, data_var):
        for i in range(data_var.shape[0]):
            for j in range(data_var.shape[1]):
                if not np.isnan(data_var[i][j]):
                    break
            if not np.isnan(data_var[i][j]):
                break
        self.assertFalse(np.isnan(data_var[i][j]), "No non-nan values in data_var")
        return i, j

class TestGoodJoins(DdiJoinBase):

    def test_join_same_everything(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=False)

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did any of the coordinates change?
        for coord_name in join.coords:
            self.assertEqual(len(join.coords[coord_name]), len(orig.coords[coord_name]), f"joined coordinate \"{coord_name}\" length does not match original coordinate length")
            self.assertTrue((join.coords[coord_name].values == orig.coords[coord_name].values).all(), f"joined coordinate \"{coord_name}\" values do not match original coordinate values")

        # did all the attributes get updated?
        for attr_name in join.attrs:
            if attr_name == 'ddi':
                continue
            self.assertEqual(join.attrs[attr_name], orig.attrs[attr_name], f"attribute values for \"{attr_name}\" do not match between original and joined")

    def test_join_different_coords(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the time dimension get merged correctly?
        orig_times = set(orig.time.values)
        cppy_times = set(cppy.time.values)
        join_times = set(join.time.values)
        self.assertEqual(len(orig_times), len(orig.time), "sets of times are missing values from their corresponding lists")
        self.assertEqual(len(cppy_times), len(cppy.time), "sets of times are missing values from their corresponding lists")
        self.assertEqual(len(join_times.difference(orig_times).difference(cppy_times)), 0, "ERROR: there are extra values in the joint times")
        self.assertEqual(len(orig.time)*2, len(join.time), "unexpected number of values in joint times")
        self.assertEqual(len(cppy.time)*2, len(join.time), "unexpected number of values in joint times")

        # did any of the other coordinates change?
        for coord_name in join.coords:
            if coord_name == 'time':
                continue
            if 'time' in join.coords[coord_name].dims:
                self.assertEqual(len(join.coords[coord_name]), len(orig.coords[coord_name])*2, f"joined coordinate \"{coord_name}\" length does not match original coordinate length x2")
                join_coords = join.coords[coord_name].sel(time=orig.time) # limit values to those from the original time coordinate
                self.assertTrue((join_coords == orig.coords[coord_name].values).all(), f"joined coordinate \"{coord_name}\" values do not match original coordinate values")
            else:
                self.assertEqual(len(join.coords[coord_name]), len(orig.coords[coord_name]), f"joined coordinate \"{coord_name}\" length does not match original coordinate length")
                self.assertTrue((join.coords[coord_name].values == orig.coords[coord_name].values).all(), f"joined coordinate \"{coord_name}\" values do not match original coordinate values")

    def test_join_different_coords_inputs_unchanged(self):
        """ this is a special test to verify that ddi_join does not modify the inputs """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)

        # set an extra attribute on vis0 (orig)
        orig.attrs['testing_extra_attr'] = 'foo'

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the attribute get carried over
        self.assertTrue('testing_extra_attr' in orig.attrs,  "vis0 should have an attribute \"testing_extra_attr\"")
        self.assertEqual(orig.testing_extra_attr, 'foo',     "vis0 should have an attribute \"testing_extra_attr\" with the value \"foo\"")
        self.assertFalse('testing_extra_attr' in cppy.attrs, "vis1 should NOT have an attribute \"testing_extra_attr\"")
        self.assertTrue('testing_extra_attr' in join.attrs,  "join should have an attribute \"testing_extra_attr\"")
        self.assertEqual(join.testing_extra_attr, 'foo',     "join should have an attribute \"testing_extra_attr\" with the value \"foo\"")

    def test_data_vars_offset_coords(self):
        """ Verify that both data_var values from time 0 and time 1 are merged in """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)
        time_len = len(orig.time)

        # set some values for the copy so that we can verify the values took
        cppy.EXPOSURE.load()
        i, j = self.helper_get_nonnan_index(cppy.EXPOSURE)
        cppy.EXPOSURE[i][j] += 1   # data variable to be merged

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the EXPOSURE data variables get updated?
        self.assertAlmostEqual(join.EXPOSURE[i+time_len][j].values + 0, orig.EXPOSURE[i][j].values + 1, 5, "unexpected value in joined EXPOSURE data variable")

    def test_join_different_attrs(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        # set some values for the copy so that we can verify the values took
        cppy.attrs['num_chan'] += 1 # attribute to be overwritten with value from orig
        cppy.attrs['ddi'] += 1      # special attribute to be combined

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the attributes get updated?
        self.assertNotEqual(orig.ddi, cppy.ddi, "original and copy should have different \"ddi\" values")
        self.assertTrue(str(orig.ddi) in join.ddi, "could not find orig's \"ddi\" value in join's \"ddi\" value")
        self.assertTrue(str(cppy.ddi) in join.ddi, "could not find cppy's \"ddi\" value in join's \"ddi\" value")
        self.assertEqual(join.num_chan, orig.num_chan, "bad value for num_chan, should have been overwritten with value from orig")

    def test_join_base(self):

        #############
        ### setup ###

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        ####################
        ### do the merge ###

        join = cngi.vis.ddijoin(orig, cppy)

        ############################
        ### validate the results ###

        # did the time dimension get merged correctly?
        orig_times = set(orig.time.values)
        cppy_times = set(cppy.time.values)
        join_times = set(join.time.values)
        self.assertEqual(len(orig_times), len(orig.time), "sets of times are missing values from their corresponding lists")
        self.assertEqual(len(cppy_times), len(cppy.time), "sets of times are missing values from their corresponding lists")
        self.assertEqual(len(join_times.difference(orig_times).difference(cppy_times)), 0, "ERROR: there are extra values in the joint times")
        self.assertEqual(len(orig.time)*2, len(join.time), "unexpected number of values in joint times")

        # did any of the other coordinates change?
        for coord_name in join.coords:
            if coord_name == 'time':
                continue
            if 'time' in join.coords[coord_name].dims:
                self.assertEqual(len(join.coords[coord_name]), len(orig.coords[coord_name])*2, f"joined coordinate \"{coord_name}\" length does not match original coordinate length x2")
                join_coords = join.coords[coord_name].sel(time=orig.time) # limit values to those from the original time coordinate
                self.assertTrue((join_coords == orig.coords[coord_name].values).all(), f"joined coordinate \"{coord_name}\" values do not match original coordinate values")
            else:
                self.assertEqual(len(join.coords[coord_name]), len(orig.coords[coord_name]), f"joined coordinate \"{coord_name}\" length does not match original coordinate length")
                self.assertTrue((join.coords[coord_name].values == orig.coords[coord_name].values).all(), f"joined coordinate \"{coord_name}\" values do not match original coordinate values")

        # did all of the data variables get updated?
        for data_var_name in join.data_vars:
            self.assertEqual(len(orig.data_vars[data_var_name])*2, len(join.data_vars[data_var_name]), f"unexpected number of values in joined data var \"{data_var_name}\"")

        # did all the attributes get updated?
        for attr_name in join.attrs:
            if attr_name == 'ddi':
                self.assertTrue(str(orig.ddi) in join.ddi, "could not find orig's \"ddi\" value in join's \"ddi\" value")
                self.assertTrue(str(cppy.ddi) in join.ddi, "could not find cppy's \"ddi\" value in join's \"ddi\" value")
            else:
                self.assertEqual(join.attrs[attr_name], orig.attrs[attr_name], f"attribute values for \"{attr_name}\" do not match between original and joined")

    def test_extra_attr_vis0(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)

        # set an extra attribute on vis0 (orig)
        orig.attrs['testing_extra_attr'] = 'foo'

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the attribute get carried over
        self.assertTrue('testing_extra_attr' in orig.attrs,  "vis0 should have an attribute \"testing_extra_attr\"")
        self.assertEqual(orig.testing_extra_attr, 'foo',     "vis0 should have an attribute \"testing_extra_attr\" with the value \"foo\"")
        self.assertFalse('testing_extra_attr' in cppy.attrs, "vis1 should NOT have an attribute \"testing_extra_attr\"")
        self.assertTrue('testing_extra_attr' in join.attrs,  "join should have an attribute \"testing_extra_attr\"")
        self.assertEqual(join.testing_extra_attr, 'foo',     "join should have an attribute \"testing_extra_attr\" with the value \"foo\"")

    def test_extra_attr_vis1(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)
        self.assertFalse('testing_extra_attr' in orig.attrs, "vis0 should NOT have an attribute \"testing_extra_attr\"")

        # set an extra attribute on vis1 (cppy)
        cppy.attrs['testing_extra_attr'] = 'foo'

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)
        print(type(join))

        # did the attribute get carried over
        self.assertTrue('testing_extra_attr' in cppy.attrs,  "vis1 should have an attribute \"testing_extra_attr\"")
        self.assertEqual(cppy.testing_extra_attr, 'foo',     "vis1 should have an attribute \"testing_extra_attr\" with the value \"foo\"")
        self.assertFalse('testing_extra_attr' in orig.attrs, "vis0 should NOT have an attribute \"testing_extra_attr\"")
        self.assertTrue('testing_extra_attr' in join.attrs,  "join should have an attribute \"testing_extra_attr\"")
        self.assertEqual(join.testing_extra_attr, 'foo',     "join should have an attribute \"testing_extra_attr\" with the value \"foo\"")

    def test_missing_nondim_coord_vis0(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        # add an extra non-dimension coordinate to the copy that is not in the original
        cppy, parent_dim_len = self.helper_add_nondim_coord(cppy, "new_coord")

        # try to merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the extra non-dimension coordinate come through?
        self.assertFalse("new_coord" in orig.coords, "Dataset \"orig\" should NOT have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in cppy.coords, "Dataset \"cppy\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.coords, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertEqual(len(join.coords["new_coord"]), parent_dim_len, "New \"new_coord\" coordinate length mismatch!")

    def test_missing_nondim_coord_vis1(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)

        # add an extra non-dimension coordinate to the original that is not in the copy
        orig, parent_dim_len = self.helper_add_nondim_coord(orig, "new_coord")

        # try to merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the extra non-dimension coordinate come through?
        self.assertFalse("new_coord" in cppy.coords, "Dataset \"orig\" should NOT have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in orig.coords, "Dataset \"cppy\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.coords, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertEqual(len(join.coords["new_coord"]), parent_dim_len, "New \"new_coord\" coordinate length mismatch!")

    def test_missing_dim_coord_vis0(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        # add an extra dimension coordinate to the copy that is not in the original
        cppy, dim_len = self.helper_add_dim_coord(cppy, "new_coord", length=4)

        # try to merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the extra non-dimension coordinate come through?
        self.assertFalse("new_coord" in orig.coords, "Dataset \"orig\" should NOT have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in cppy.coords, "Dataset \"cppy\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.coords, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.dims, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertEqual(len(join.coords["new_coord"]), dim_len, "New \"new_coord\" coordinate length mismatch!")

    def test_missing_dim_coord_vis1(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)

        # add an extra dimension coordinate to the original that is not in the copy
        orig, dim_len = self.helper_add_dim_coord(orig, "new_coord", length=4)

        # try to merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the extra non-dimension coordinate come through?
        self.assertFalse("new_coord" in cppy.coords, "Dataset \"orig\" should NOT have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in orig.coords, "Dataset \"cppy\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.coords, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertTrue("new_coord" in join.dims, "Dataset \"join\" should have the \"new_coord\" coordinate!")
        self.assertEqual(len(join.coords["new_coord"]), dim_len, "New \"new_coord\" coordinate length mismatch!")

    def test_same_coords_one_nan_data_var(self):
        """ Two datasets should be able to be joined if the same data_var has different values at the same coordinate position, and one of those value is NaN. """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=False)

        # find a non-nan index and change the original to be a nan value
        orig.EXPOSURE.load()
        cppy.EXPOSURE.load()
        i, j = self.helper_get_nonnan_index(cppy.EXPOSURE)
        orig.EXPOSURE[i][j] = np.nan
        cppy.EXPOSURE[i][j] = 1
        self.assertNotEqual(orig.EXPOSURE[i][j], cppy.EXPOSURE[i][j])

        # do the merge
        join = cngi.vis.ddijoin(orig, cppy)

        # verify the non-nan value got copied
        self.assertTrue(np.isnan(orig.EXPOSURE[i][j]))
        self.assertEqual(float(cppy.EXPOSURE[i][j].values), 1)
        self.assertEqual(float(join.EXPOSURE[i][j].values), 1)

    def test_extra_data_var(self):
        """ Should be able to merge two datasets that are identical, except that one of them has an extra data_var that the other doesn't. """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=False)

        # add an extra data_var to the copy that is not in the original
        cppy, data_var_len = self.helper_add_data_var(cppy, "new_data_var", "time")

        # try to merge
        join = cngi.vis.ddijoin(orig, cppy)

        # did the extra data_var come through?
        self.assertFalse("new_data_var" in orig.data_vars, "Dataset \"orig\" should NOT have the \"new_data_var\" data_var!")
        self.assertTrue("new_data_var" in cppy.data_vars, "Dataset \"cppy\" should have the \"new_data_var\" data_var!")
        self.assertTrue("new_data_var" in join.data_vars, "Dataset \"join\" should have the \"new_data_var\" data_var!")
        self.assertEqual(len(join.data_vars["new_data_var"]), data_var_len, "New \"new_data_var\" data_var length mismatch!")

class TestBadJoins(DdiJoinBase):

    def test_different_nondim_coords(self):
        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=True)

        # change one of the non-dimension coordinates to be incompatible with the orig coordinate
        cppy.antennas.load()
        cppy.antennas[-1] += 1

        # try to merge
        with self.assertRaises(Exception, msg="ddi_join should not allow datasets with differing non-dimension coordinates to be merged"):
            cngi.vis.ddijoin(orig, cppy)

    def test_same_datavar_diff_coords(self):
        """ Two datasets should not be able to be joined if the same data_var has different coordinates. """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(deep_copy_both=True, cppy_change_times=True)

        # add two different dimension coordinates
        orig, dim_len = self.helper_add_dim_coord(orig, "orig_coord", length=4)
        cppy, dim_len = self.helper_add_dim_coord(cppy, "cppy_coord", length=4)

        # add the same data_var, but reference the different coordinates
        orig = orig.assign({"new_data_var": xr.DataArray([1,2,3,4], dims="orig_coord")})
        cppy = cppy.assign({"new_data_var": xr.DataArray([1,2,3,4], dims="cppy_coord")})

        # try to merge
        with self.assertRaises(Exception, msg="ddi_join should not allow datasets with the same data_var referencing different coordinates to be merged"):
            cngi.vis.ddijoin(orig, cppy)

    def test_same_coords_diff_data_var(self):
        """ Two datasets should be able to be joined if the same data_var has different values at the same coordinate position, and one of those value is NaN. """

        # get the ddis
        orig, cppy = self.helper_get_joinable_ddis(cppy_change_times=False)

        # find a nan-index
        cppy.EXPOSURE.load()
        i, j = self.helper_get_nonnan_index(cppy.EXPOSURE)

        # change one of the data_vars values in the copy
        cppy.EXPOSURE[i][j] += 1
        self.assertNotEqual(orig.EXPOSURE[i][j], cppy.EXPOSURE[i][j])

        # try to merge
        with self.assertRaises(Exception, msg="ddi_join should not allow datasets with a different value for their data_var to be merged"):
            cngi.vis.ddijoin(orig, cppy)


