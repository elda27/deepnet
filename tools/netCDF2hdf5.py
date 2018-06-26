import xarray
import sys

xarray.open_dataset(sys.argv[1]).to_dataframe().reset_index().to_hdf(sys.argv[2], 'metric')