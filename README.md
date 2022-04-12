gr Predictor 
====

"gr Predictor" is a deep-learning model rapidly estimating the water-oxygen distribution function around protein obtained by the 3D-RISM theory from the protein 3D structure. The computation is completed using either a single CPU (a few minutes) or a single GPU (tens of second).

## Requirement
Python 3.6~, Anaconda, openbabel, dask, tesorflow

## License
“gr Predictor” is available under the GNU General Public License.

## Citing this work
If you use “gr Predictor”, please cite:

```
gr Predictor: a Deep-Learning Model for Predicting the Hydration Structures around Proteins 
Kosuke Kawama, Yusaku Fukushima, Mitsunori Ikeguchi, Masateru Ohta, and Takashi Yoshidome
Submitted.
```

## Usage

* Import modules

  `import guv_prediction as gpred`
  
* Prediction of g(r)

  `spliter = gpred.Prediction( pdb_dir, dx_dir, pred_area_center = np.array([x, y, z]), pred_area_range = w )`

  * `pdb_dir` : Path to the PDB file      
  
  * `dx_dir` : Path to the output file (dx format)
  
  * `pred_area_center` : Central coordinate of the prediction region in the protein ( x, y, and z [angstrom] ) 

  * `pred_area_range` : Width of the prediction region ( w [angstrom] )    
