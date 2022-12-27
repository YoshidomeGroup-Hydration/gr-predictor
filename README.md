gr Predictor 
====

"gr Predictor" is a deep-learning model rapidly estimating the water-oxygen distribution function around protein obtained by the 3D-RISM theory from the protein 3D structure. The computation is completed within a few minutes when using a single CPU and approximately one minute when using a single GPU.

## Requirement
Python 3.6~, Anaconda, ~~openbabel~~, dask, tesorflow  
**New! The "gr Predictor" is now available on the Google Colaboratory. (2022/12/27)**

## License
“gr Predictor” is available under the GNU General Public License.

## Citing this work
If you use “gr Predictor”, please cite:

```
gr Predictor: a Deep-Learning Model for Predicting the Hydration Structures around Proteins 
Kosuke Kawama, Yusaku Fukushima, Mitsunori Ikeguchi, Masateru Ohta, and Takashi Yoshidome
Journal of Chemical Information and Modeling, Vol. 62, 4460 (2022).
```
## Contact
If you have any questions, please contact Takashi Yoshidome at takashi.yoshidome.b1@tohoku.ac.jp.

## Usage

* Import modules

  `import guv_prediction as gpred`
  
* Prediction of g(r)

  `spliter = gpred.Prediction( pdb_dir, dx_dir, pred_area_center = np.array([x, y, z]), pred_area_range = w )`

  * `pdb_dir` : Path to the PDB file      
  
  * `dx_dir` : Path to the output file of the water-oxygen distribution function around a protein (dx format)
  
  * `pred_area_center` : Central coordinate of the prediction region in the protein ( x, y, and z [angstrom] ) 

  * `pred_area_range` : Width of the prediction region ( w [angstrom] )    

## Usage for Google Colaboratory
1. Select GPU from the Runtime tab.

2. Execute the following codes.  
	`!git clone https://github.com/YoshidomeGroup-Hydration/gr-predictor.git`  
	`%cd /content/gr-predictor`  
	`import guv_prediction_f as gpred`  
	`%cd ../`

3. Upload a PDB file (PDB.pdb) to /content/. Add hydrogens to the protein in advance.
	
4. Execute the following code. 
	`spliter = gpred.Prediction("PDB.pdb", "PDB_pred.dx", model_dir="/content/gr-predictor/model_1.h5")`

5. The water-oxygen distribution function is output to /content/PDB_pred.dx. Please download the dx file. This file can be seen using chimera or vmd.
