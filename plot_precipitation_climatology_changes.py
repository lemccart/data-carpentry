import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean

import pdb

def convert_pr_units(darray):
    """Convert kg m-2 s-1 to mm day-1.

    Args:
      darray (xarray.DataArray): Precipitation data

    """
    assert darray.units == 'kg m-2 s-1', "Program assumes input units are kg m-2 s-1"
    
    darray.data = darray.data * 86400
    darray.attrs['units'] = 'mm/day'

    return darray


def create_plot(clim, model, season, gridlines=False, levels=None):
    """Plot the precipitation climatology.

    Args:
      clim (xarray.DataArray): Precipitation climatology data
      model (str): Name of the climate model
      season (str): Season

    Kwargs:
      gridlines (bool): Select whether to plot gridlines
      colorbar_levels (list): option to add customized colorbar ticks

    """

    if not levels:
        levels = np.arange(0., 13., 1.)

    fig = plt.figure(figsize=[15,7])

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    clim.sel(season=season).plot.contourf(ax=ax,
                                          levels=levels,
                                          extend='max',
                                          transform=ccrs.PlateCarree(),
                                          cbar_kwargs={'label': clim.units},
                                          cmap=cmocean.cm.haline_r)
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()

    title = f'{model} precipitation climatology ({season})'
    plt.title(title)

def apply_mask(darray, sftlf_file, realm):
    """Mask ocean or land using a sftlf (land surface fraction) file.

    Args:
     darray (xarray.DataArray): Data to mask
     sftlf_file (str): Land surface fraction file
     realm (str): Realm to mask

    """

    dset = xr.open_dataset(sftlf_file)
    #pdb.set_trace()
    assert realm in ['land', 'ocean'], """Valid realms are 'land' or 'ocean'"""

    if realm == 'land':
        masked_darray = darray.where(dset['sftlf'].data < 50)
    else:
        masked_darray = darray.where(dset['sftlf'].data > 50)

    return masked_darray

def main(inargs):
    """Run the program."""

    dset = xr.open_dataset(inargs.pr_file)
    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim)

    if inargs.mask:
        sftlf_file, realm = inargs.mask
        clim = apply_mask(clim, sftlf_file, realm)

    create_plot(clim, dset.attrs['source_id'], inargs.season, gridlines=inargs.gridlines, levels=inargs.colorbar_levels)
    plt.savefig(inargs.output_file, dpi=200)


if __name__ == '__main__':
    description='Plot the precipitation climatology for a given season.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("season", type=str, choices=['DJF', 'MAM', 'JJA', 'SON'], help="Season to plot")
    parser.add_argument("output_file", type=str, help="Output file name")
    parser.add_argument("--gridlines", action="store_true", default=False, help="Option to include gridlines")
    parser.add_argument("--colorbar_levels", type=float, nargs='*', default=None, help="Option to include custom colorbar ticks")
    parser.add_argument("--mask", type=str, nargs=2, metavar=('SFTLF_FILE', 'REALM'), default=None, help="""Provide sftlf file and realm to mask ('land' or 'ocean')""")

    args = parser.parse_args()

    main(args)
