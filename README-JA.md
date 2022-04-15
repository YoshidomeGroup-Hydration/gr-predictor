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
## Contact
If you have any questions, please contact Takashi Yoshidome at takashi.yoshidome.b1@tohoku.ac.jp.

## Usage
* モジュールのインポート

  `import guv_prediction as gpred`
  
* 予測の実行

  `spliter = gpred.Prediction( pdb_dir, dx_dir, pred_area_center = np.array([x, y, z]), pred_area_range = w )`

  * `pdb_dir` : PDBファイルのパス      
  
  * `dx_dir` : 出力するdxファイルのパス
  
  * `pred_area_center` : 予測領域の中心座標 ( x,y,z [Å] ) 

  * `pred_area_range` : 予測領域の幅 ( w [Å] )    

## Usage for lab
  

* サブオプション一覧（研究室用）
  
  * `guv_dir` : guvファイルのパス。
  * `split_center` : ロスを最小化した部分の大きさ。デフォルトは 16 [voxel]
  * `split_size` : 分割する箱の大きさ。デフォルトは48 [voxel]
 
* クラス内の変数解説
  * spliter.g_pred : 予測g 
  * spliter.g_true : 正解g (guvファイルが与えられたときのみ生成)
  * spliter.g_true_local : 指定した局所領域における正解g　(guvファイルが与えられたときのみ生成)
  * spliter.g_pred_for_compare : 正解gと比較する用の局所領域における予測g。（予測する局所領域が3D-RISMの作るグリッド空間からはみ出す場合、正解gとの比較のためにはみ出した空間を予測gから切り出す。spliter.g_true_localと同じ形状になる）
  * 
