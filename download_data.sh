# shell commands to download the public sdss boss large scale structures catalogs used for our example magnification bias estimates
# you will need to go into the folder and unzip the files after downloading

mkdir SDSS_DR12

#SDSS
wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASS_North.fits.gz
wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASS_South.fits.gz

#to extract gz files on mac can use
cd SDSS_DR12
gunzip galaxy_DR12v5_CMASS_North.fits.gz
gunzip galaxy_DR12v5_CMASS_South.fits.gz
cd ..

#(uncomment the others to download as well)
## LOWZ 
#wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_North.fits.gz
#wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_South.fits.gz

## z1 and z3
#wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz
#wget -P SDSS_DR12/ https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz
