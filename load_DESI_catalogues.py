import os
import numpy as np
import healpy as hp
from astropy.table import Table, join, vstack, Column
from astropy.io import fits
import re

class Version:
    def __init__(self, version):
        self.numeric_parts, self.suffix = self.parse_version(version)
    
    @staticmethod
    def parse_version(version):
        """
        Parse the version string into numeric parts and a suffix.

        Args:
        version (str): The version string (e.g., 'v1.2.3pip').

        Returns:
        tuple: A tuple containing a list of integers for the numeric parts and a suffix string.
        """
        match = re.match(r'v?([\d.]+)([a-zA-Z]*)', version)
        numeric_parts = list(map(int, match.group(1).split('.')))
        suffix = match.group(2)
        return numeric_parts, suffix
    
    def __lt__(self, other):
        for i in range(max(len(self.numeric_parts), len(other.numeric_parts))):
            self_part = self.numeric_parts[i] if i < len(self.numeric_parts) else 0
            other_part = other.numeric_parts[i] if i < len(other.numeric_parts) else 0
            if self_part != other_part:
                return self_part < other_part
        return self.suffix < other.suffix
    
    def __gt__(self, other):
        for i in range(max(len(self.numeric_parts), len(other.numeric_parts))):
            self_part = self.numeric_parts[i] if i < len(self.numeric_parts) else 0
            other_part = other.numeric_parts[i] if i < len(other.numeric_parts) else 0
            if self_part != other_part:
                return self_part > other_part
        return self.suffix > other.suffix
    
    def __eq__(self, other):
        return self.numeric_parts == other.numeric_parts and self.suffix == other.suffix
    
    def __le__(self, other):
        return self < other or self == other
    
    def __ge__(self, other):
        return self > other or self == other
    
    def __str__(self):
        return f"v{'.'.join(map(str, self.numeric_parts))}{self.suffix}"


def assign_systematic_property(lenstable,galaxy_type,sysname,version,nside=256,
                              fname = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/hpmaps/{}_mapprops_healpix_nested_nside256.fits"):
    hpmapsdir = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/hpmaps/"
    if not os.path.exists(hpmapsdir.format(version)):
        print(f"{version} does not have hpmaps in {hpmapsdir.format(version)}")
        available_versions = [Version(x) for x in os.listdir("/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/") if x.startswith("v") and os.path.isdir("/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/"+x)]
        available_versions = np.sort(available_versions)
        mask = available_versions <= version
        available_versions = available_versions[mask]
        for _version in available_versions[::-1]:
            # see if the hpmaps directory exists
            if os.path.exists(hpmapsdir.format(version)):
                print(f"Using hpmaps from {_version}")
                break
    else:
        _version = version

    fname_read = fname.format(_version,galaxy_type)


    try:
        systable = Table.read(fname_read)
    except Exception as e:
        import warnings
        warnings.warn("Systable not found, searching for north and south tables.")
        fname_read = fname_read.split(".fits")[0]+"_{}.fits"
        result = np.zeros(len(lenstable))
        for reg in ["N","S"]:
            systable = Table.read(fname_read.format(reg))
            mask = (lenstable["PHOTSYS"] == reg)
            phi,theta = np.radians(lenstable['RA'][mask]),np.radians(90.-lenstable['DEC'][mask])
            ipix = hp.ang2pix(nside,theta,phi,nest=True)
            result[mask] = systable[sysname][ipix]
        return result

    phi,theta = np.radians(lenstable['RA']),np.radians(90.-lenstable['DEC'])
    ipix = hp.ang2pix(nside,theta,phi,nest=True)
    return systable[sysname][ipix]

def hdul_to_table(hdul,columns=None):
    data = Table()
    if columns is None:
        columns = hdul[1].columns.names
    for col in columns:
        data.add_column(hdul[1].data.field(col),name=col)
    return data

def cut_to_unique_TARGETID(table):
    # Get the TARGETID column as a NumPy array
    targetids = table['TARGETID'].data

    # Find the indices of unique TARGETIDs
    # We use 'return_index=True' to get the index of the first occurrence of each unique value
    _, unique_indices = np.unique(targetids, return_index=True)

    # Sort the indices to maintain the original order
    unique_indices.sort()

    # Create a new table with only the unique TARGETIDs
    unique_table = table[unique_indices]

    return unique_table

def read_table(filename, columns=None, memmap=True, tabulatedbool=False):
    if columns is None:
        return Table.read(filename)
    requests = {}
    with fits.open(filename,memmap=memmap) as hdul:
        not_available_columns = list(set(columns)-set(hdul[1].columns.names))
        available_columns = list(set(columns)-set(not_available_columns))
        if len(not_available_columns)>0 and not "TARGETID" in available_columns:
            # add TARGETID
            available_columns = ["TARGETID"]+available_columns
        # if it is a full map, load PHOTSYS column, in case we need to add systematics from healpix maps later
        if "_full_HPmapcut" in os.path.basename(filename):
            for key in ["RA","DEC","PHOTSYS"]:
                # load RA,DEC and PHOTSYS for assign_systematic_property
                if key not in available_columns:
                    available_columns = [key]+available_columns
                    requests[key] = True
        data = hdul_to_table(hdul,columns=available_columns)

    if len(not_available_columns)>0:
        print(f"Columns {not_available_columns} not available in file {filename}")
        if "_clustering" in os.path.basename(filename):
            # first priority try to read from full_HPmapcut file
            print("Trying to match from full_HPmapcut file")
            not_available_columns = list(set(not_available_columns)-set(['WEIGHT_FKP']))
            if "DA2" in filename:
                tab_full_HPmapcut = read_table(filename.replace("_clustering","_full_HPmapcut").replace("nonKP",""),columns=["TARGETID"]+not_available_columns,memmap=memmap)
            else:
                tab_full_HPmapcut = read_table(filename.replace("_clustering","_full_HPmapcut"),columns=["TARGETID"]+not_available_columns,memmap=memmap)

            if len(np.unique(tab_full_HPmapcut["TARGETID"]))!=len(tab_full_HPmapcut["TARGETID"]):
                print("WARNING: TARGETID not unique in full_HPmapcut file: {} vs {}".format(np.unique(len(tab_full_HPmapcut["TARGETID"])),len(tab_full_HPmapcut["TARGETID"])))
                tab_full_HPmapcut = cut_to_unique_TARGETID(tab_full_HPmapcut)

            len_before = len(data)
            data = join(data,tab_full_HPmapcut,keys="TARGETID",join_type="inner")
            len_after = len(data)
            assert len_before==len_after, f"Length mismatch: {len_before} vs {len_after}, full_HPmapcut file {len(tab_full_HPmapcut)}"
            print("Matching weights from clustering_NGC and clustering_SGC")
            #tab_NGC = Table(fits.open(filename.replace("_clustering","_NGC_clustering"),columns=['TARGETID','WEIGHT_FKP'])[1].data)
            #tab_SGC = Table(fits.open(filename.replace("_clustering","_SGC_clustering"),columns=['TARGETID','WEIGHT_FKP'])[1].data)
            hdul = fits.open(filename.replace("_clustering","_NGC_clustering"),memmap=True)
            tab_NGC = hdul_to_table(hdul,columns=['TARGETID','WEIGHT_FKP'])
            hdul = fits.open(filename.replace("_clustering","_SGC_clustering"),memmap=True)
            tab_SGC = hdul_to_table(hdul,columns=['TARGETID','WEIGHT_FKP'])

            tab_with_weights = vstack((tab_NGC, tab_SGC))
            data = join(data, tab_with_weights, keys="TARGETID",join_type="inner")
            
            if tabulatedbool:
                axis_ratio = np.zeros(len(data['RA']))
                sersic = np.zeros(len(data['RA']))
                gaia_gmag = np.zeros(len(data['RA']))
                pix = hp.ang2pix(8,data['RA'],data['DEC'],lonlat=True,nest=True)
                for ppix in np.unique(pix):
                    #ppix = np.unique(pix)[0]
                    sel = np.where(pix == ppix)
                    if "BGS" in os.path.basename(filename):
                        targ = fits.open('/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/bright/targets-bright-hp-%i.fits' % ppix)[1].data
                    else:
                        targ = fits.open('/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/dark/targets-dark-hp-%i.fits' % ppix)[1].data
                    joined_cat = join(data[sel], targ, keys='TARGETID')
                    ellipticity = (joined_cat['SHAPE_E1']**2 + joined_cat['SHAPE_E2']**2)**0.5
                    axis_ratio[sel] = (1 + ellipticity) / (1 - ellipticity)
                    sersic[sel] = joined_cat['SERSIC']
                    gaia_gmag[sel] = joined_cat['GAIA_PHOT_G_MEAN_MAG']
                    print(ppix)
                c1 = Column(axis_ratio, name='AXIS_RATIO')
                c2 = Column(sersic, name='SERSIC')
                c3 = Column(gaia_gmag,name='GAIA_GMAG')
                data.add_column(c1, index=0)
                data.add_column(c2, index=0)
                data.add_column(c3, index=0)

        else:
            # assign via assign_systematic_property function
            if "WEIGHT" in not_available_columns:
                pass
            else:
                print(f"Assigning {not_available_columns} from healpix maps")
                galaxy_type = os.path.basename(filename).split("_")[0]
                if galaxy_type == 'ELG':
                    galaxy_type = 'ELG_LOPnotqso'
                if galaxy_type == "BGS":
                    galaxy_type = "BGS_BRIGHT"
                version = filename.split("/v1.")[1]
                version = version.split("/")[0]
                version = Version("v1."+version)
                for col in not_available_columns:
                    data[col] = assign_systematic_property(data,galaxy_type,col,version)
    assert not is_table_masked(data), f"Table {filename} is masked"
    # remove all columns that were requested
    for key in requests.keys():
        data.remove_column(key)
    return data


def is_table_masked(table):
    return any(getattr(col, 'mask', None) is not None for col in table.columns.values())