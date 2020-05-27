import pytest
import os
from cosima_cookbook.utils import stats
import numpy as np
import xarray as xr

def linear_trend_noise(noise_scale,n,trend_scale=1):
    """
    Create simple trend with noise
    """
    # Dimensions of dataarray
    time = np.arange(n)
    x = np.arange(4)
    y = np.arange(4)

    data = np.zeros((len(time), len(x), len(y)))

    da = xr.DataArray(data, coords=[time, x , y], 
                        dims=['time', 'lon', 'lat'])

    noise = np.random.randn(*np.shape(data))
    linear_trend = xr.DataArray(time, coords=[time], dims=['time'])

    da_with_linear_trend = (da + linear_trend * trend_scale) + noise * noise_scale
    return da_with_linear_trend
    
n=100
exps=np.array(list(zip(np.arange(0,n/2,5),n+np.arange(0,n/2,5)*0)))
@pytest.mark.parametrize(('noise_scale','n'), [*exps])
def test_significance(noise_scale,n):
    da = linear_trend_noise(noise_scale,n)
    MK_class = stats.Mann_Kendall_test(da, 'time')
    MK_trends = MK_class.compute()
    assert(MK_trends.signif.values.all() == 1)

n=100
exps=np.array(list(zip(np.arange(0,n/2,5)*0,n+np.arange(0,n/2,5)*0,np.arange(-0.70,0.65,0.15))))
@pytest.mark.parametrize(('noise_scale','n','trend_scale'), [*exps])
def test_slope(noise_scale,n,trend_scale):
    # Assess the value of the slope similar to the given one
    da = linear_trend_noise(noise_scale,n,trend_scale)
    MK_class = stats.Mann_Kendall_test(da, 'time')
    MK_trends = MK_class.compute()
    assert(np.isclose(MK_trends.trend.values, trend_scale).all())

def test_slope_methods():
    # Assert if simple slopes are similar.
    da = linear_trend_noise(0,100,0.4)
    MK_class_lr = stats.Mann_Kendall_test(da, 'time',method='linregress')
    MK_trends_lr = MK_class_lr.compute(path='./tmp.nc')
    MK_class_ts = stats.Mann_Kendall_test(da, 'time',method='theilslopes')
    MK_trends_ts = MK_class_ts.compute(save = True)

    os.remove('./tmp.nc')
    assert(np.isclose(MK_trends_lr.trend.values, MK_trends_ts.trend.values).all())

def test_slope_modified_MK():
    # Assert if correlated data is significant.
    n=500
    da = linear_trend_noise(0,n,1/5)
    time=np.arange(n)
    sin = xr.DataArray(np.sin(time)*10, coords=[time], dims=['time'])

    MK_class = stats.Mann_Kendall_test(da+sin, 'time',MK_modified=True,alpha=0.1)
    MK_trends = MK_class.compute()

    assert(MK_trends.signif.values.all() == 1)


def test_slope_modified_MK_large_cycle():
    # Assert if correlated data is significant.
    n=500
    da = linear_trend_noise(0,n,1/5)
    time=np.arange(n)
    sin = xr.DataArray(np.sin(time)*100, coords=[time], dims=['time'])

    MK_class = stats.Mann_Kendall_test(da+sin, 'time',MK_modified=True,alpha=0.1)
    MK_trends = MK_class.compute()

    assert(MK_trends.signif.values.all() == 0)


def test_slope_methods_fail():
    # Assert if simple slopes are similar.
    da = linear_trend_noise(0,100,0.4)
    MK_class_ts = stats.Mann_Kendall_test(da, 'time',method='')
    with pytest.raises(ValueError):
        MK_trends_ts = MK_class_ts.compute() 

def test_slope_with_nan():
    # Assert if function works with nan values.
    n=100
    da = linear_trend_noise(0,n,1/5)
    da_masked = da.where(~np.logical_and(da.lon<2,da.lat>2))

    MK_class = stats.Mann_Kendall_test(da_masked, 'time',alpha=0.1)
    MK_trends = MK_class.compute()

    values=MK_trends.signif.values

    assert(values[np.isfinite(MK_trends.trend.values)].all() == 1)

def test_slope_with_nan_MK_modif():
    # Assert if function works with nan values.
    n=100
    da = linear_trend_noise(0,n,1/5)
    da_masked = da.where(~np.logical_and(da.lon<2,da.lat>2))

    MK_class = stats.Mann_Kendall_test(da_masked, 'time', MK_modified=True, alpha=0.1)
    MK_trends = MK_class.compute()

    values=MK_trends.signif.values
    assert(values[np.isfinite(MK_trends.trend.values)].all() == 1)


def test_no_slope():
    # Assert if function works with nan values.
    n=100
    da = linear_trend_noise(0,n,1/5)*0

    MK_class = stats.Mann_Kendall_test(da, 'time', alpha=0.1)
    MK_trends = MK_class.compute()

    assert(MK_trends.signif.values.all() == MK_class.score)


def test_rename():
    # Assert if function works with nan values.
    n=100
    da = linear_trend_noise(0,n,1/5).rename({'lon':'x','lat':'y'})
    MK_class = stats.Mann_Kendall_test(da, 'time', alpha=0.1,coords_name={'lon':'x','lat':'y'})
    MK_trends = MK_class.compute()

def test_main():
    stats.init('__main__')