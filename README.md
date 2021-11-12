$ g(\textbf{r}) $ Predicter 
====

## Requirement
Python 3.6~, Anaconda, openbabel, dask, tesorflow

## Usage

* モジュールのインポート

  `import guv_prediction as gpred`
  
* 予測の実行

  `spliter = gpred.Prediction( pdb_dir, dx_dir, pred_area_center = np.array([x, y, z]), pred_area_range = w )`

  * pdb_dir : PDBファイルのパス      
  
  * dx_dir : 出力するdxファイルのパス
  
  * pred_area_center : 予測領域の中心座標 ( $x,y,z$ [$Å$] ) 

  * pred_area_range : 予測領域の幅 ( $w$ [$Å$] )     
  