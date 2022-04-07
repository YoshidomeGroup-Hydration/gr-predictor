gr Predicter 
====

## Requirement
Python 3.6~, Anaconda, openbabel, dask, tesorflow

## Usage

* Import modules

  `import guv_prediction as gpred`
  
* Prediction of g(r)

  `spliter = gpred.Prediction( pdb_dir, dx_dir, pred_area_center = np.array([x, y, z]), pred_area_range = w )`

  * `pdb_dir` : Path to PDB file      
  
  * `dx_dir` : Path to output file (dx format)
  
  * `pred_area_center` : Central coordinate of the prediction region in the protein ( x, y, and z [angstrom] ) 

  * `pred_area_range` : Width of the prediction region ( w [angstrom] )    

* 

