
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import time
import json
import pandas as pd
import numpy as np
from openbabel import openbabel as ob
from numba import jit, njit, prange
import dask.dataframe as dd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
def lmse(y_true, y_pred):
    mses = mse(y_true[:,16:32,16:32,16:32,:], y_pred[:,16:32,16:32,16:32,:])
    return K.sum(mses)

class Prediction():
    """
    水予測
    """
    def __init__(self, pdb_dir, dx_dir, split_center=16, split_size=48, guv_dir=None, pred_area_center=np.array([None]), pred_area_range=-1):
        
        self.pdb_dir = pdb_dir
        if self.pdb_dir==None:print("Error ; PDB file dir")
        self.atype, self.Atom_x, self.Atom_y, self.Atom_z = self.getAtoms(ion=False)
        
        self.analysis_dir = guv_dir
        self.n = None
        self.R_Grid = None
        
        if self.analysis_dir == None:
            # オリジナルの座標基準値を生成
            min_ax = (np.array([self.Atom_x.min(), self.Atom_y.min(), self.Atom_z.min()])-14.).astype(np.int32)
            max_ax = (np.array([self.Atom_x.max(), self.Atom_y.max(), self.Atom_z.max()])+14.).astype(np.int32)
            self.R_Grid = min_ax.astype(np.float64)
            self.n = ((max_ax-min_ax)*2).astype(np.int32)
    
        else:
            # 3DRISMの座標基準値を使用
            self.analysis = dd.read_csv(self.analysis_dir, header = None, delim_whitespace = True).compute()
            self.R_Grid = np.array(self.analysis.min())[0:3]
            print("origin R_Grid : ",self.R_Grid)
            self.getGrid()
            mizvoxel = self.mizPerform()
            #self.g_true = mizvoxel.reshape((1,) + mizvoxel.shape)
            self.g_true = mizvoxel
            
        self.center = split_center # 箱の中心の大きさ
        self.split_size = split_size # 箱の大きさ
        
        self.pred_area_center = pred_area_center
        self.pred_area_range = pred_area_range
        
        self.g_pred = None
        self.g_true_local = None
        self.g_pred_for_compare = None
        self.provoxel = None
        
        # 局所座標が与えられた場合
        if pred_area_center[0] != None:
            if pred_area_range < 0:
                print("ERROR : Invalid value of range")
                return None
            self.pred_area_center = pred_area_center
            self.pred_area_range = pred_area_range
            # 局所予測
            self.predicter_local(model_dir = "model_1.h5")
        # 通常ver
        else:
            # 全体ボクセル化
            self.provoxel = self.proPerform()
            self.lost = None # 作業用　RISMのグリッドサイズからどれだけ削ったor増した箱を作ったか記録する配列
            self.x_range, self.x_range, self.x_range = None, None, None  # 作業用　分割時のパラメータの記録
            self.hakosize = None #作業用　define_hakoで求めた箱の大きさの記録
            self.predicter_normal(model_dir = "model_1.h5")
            
        if dx_dir != None:
            self.to_dx(self.g_pred, dx_dir)
        
    # 3DRISMのボックスの大きさを計算する 単位はボクセル
    def getGrid(self):
        my = lambda x: np.round((x * 2 + 1) // 2)
        self.n = np.array([my((self.analysis[0] - self.R_Grid[0])*2).astype(np.int64).max()+1, my((self.analysis[1] - self.R_Grid[1])*2).astype(np.int64).max()+1, my((self.analysis[2] - self.R_Grid[2])*2).astype(np.int64).max()+1])
        
    # タンパクをボクセル化
    def proPerform(self):
        
        voxel = self.proVoxelizer_Ion(self.atype.astype(np.int32), self.Atom_x, self.Atom_y, self.Atom_z, self.R_Grid, self.n.astype(np.int64),np.zeros([3,2]))
        return voxel

    # pandasで読み込んだxyzvをnumpyにして返す                
    def mizPerform(self):
        
        my = lambda x: np.round((x * 2 + 1) // 2)
        self.analysis[0], self.analysis[1], self.analysis[2] = my((self.analysis[0] - self.R_Grid[0])*2).astype(int), my((self.analysis[1] - self.R_Grid[1])*2).astype(int), my((self.analysis[2] - self.R_Grid[2])*2).astype(int)
        a = self.analysis.values
        x, y, z = a[:,0].astype(int), a[:,1].astype(int), a[:,2].astype(int)
        voxel = np.zeros((self.n[0], self.n[1], self.n[2]))
        voxel[x, y, z] = self.analysis[3]
        return voxel
    
    # pdbを読み込む。ライブラリopenbabelが必要
    # ↓の自作(getAtoms_new)が使えないときの代わり
    # 返り値：　atype : タイプを原子数分いれた配列
    # 　　　　　Atom_* : *座標を原子数分いれた配列
    # ion=Trueで[CNOSH]以外の原子の座標のリストと、[CNOSH]原子の原子座標のリストを返す
    def getAtoms(self, ion=False): 
        
        obConversion = ob.OBConversion()
        mol = ob.OBMol()
        obConversion.ReadFile(mol, self.pdb_dir)
        atype = [obatom.GetAtomicNum() for obatom in ob.OBMolAtomIter(mol)]
        Atom_x = [obatom.GetX() for obatom in ob.OBMolAtomIter(mol)]
        Atom_y = [obatom.GetY() for obatom in ob.OBMolAtomIter(mol)]
        Atom_z = [obatom.GetZ() for obatom in ob.OBMolAtomIter(mol)]
        if ion:
            def check_hete(atomicnum):
                if atomicnum!=1 and atomicnum!=6 and atomicnum!=7 and atomicnum!=8 and atomicnum!=16 : return True
                else:return False
            Atom_hete = np.array([np.array([obatom.GetX(), obatom.GetY(), obatom.GetZ()]) for obatom in ob.OBMolAtomIter(mol) if check_hete(obatom.GetAtomicNum())])
            Atom = np.array([np.array([obatom.GetX(), obatom.GetY(), obatom.GetZ()]) for obatom in ob.OBMolAtomIter(mol) if not check_hete(obatom.GetAtomicNum())])

        obConversion.CloseOutFile()
        if ion:return np.array(atype), np.array(Atom_x), np.array(Atom_y), np.array(Atom_z), Atom_hete, Atom
        else:return np.array(atype), np.array(Atom_x), np.array(Atom_y), np.array(Atom_z)
            
    
    # タンパクをボクセル化してnpyにして返す
    # atype : タイプを原子数分いれた配列
    # Atom_* : *座標を原子数分いれた配列
    # R_Grid : 3DRISMで作った箱の座標の最小値 [x_min,y_min,z_min]
    # n : 3DRISMで作った箱の大きさ 単位はボクセル [xsize, ysize, zsize]
    # jitライブラリで計算　CPUを全部使う
    @staticmethod
    @njit('f8[:,:,:,:](i4[:,], f8[:,], f8[:,], f8[:,], f8[:,], i8[:,], f8[:,:,])', parallel=True, cache=True)
    def proVoxelizer_Ion(atype, Atom_x, Atom_y, Atom_z, R_Grid, n, local_area):
        
        # 各原子タイプにおけるファンデルワールス半径の定義
        van = {6:1.69984, 7:1.62500, 8:1.51369, 16:1.78180, 1:1.2, 30:0.97999, 20:1.525, 12:0.706125}
        # 各原子タイプをしまうチャンネル次元のインデックスを決定。デフォルトの順番は[炭素,窒素,酸素,硫黄,水素,亜鉛,カルシウム,マグネシウム]
        atm = {6:0, 7:1, 8:2, 16:3, 1:4, 30:5, 20:6, 12:7}
        
        # 配列の初期化
        voxel = np.zeros((8,n[0],n[1],n[2]))
        
        # 原子, 原子座標-5~+5ボクセルの範囲でfor文を回し、ボクセル値を決定
        for i in prange(Atom_x.shape[0]):
            
            xg, yg, zg =  Atom_x[i], Atom_y[i], Atom_z[i]
            if local_area.max() > 0:
                if local_area[0,0] > xg or local_area[0,1] < xg or \
                   local_area[1,0] > yg or local_area[1,1] < yg or \
                   local_area[2,0] > zg or local_area[2,1] < zg:
                    continue
                
            # i番目の原子のx,y,z座標から3DRIMSの基準（各次元の最小値）を引き、i番目の原子が存在するグリッド位置を算出
            kyori = np.array([xg, yg, zg]) - R_Grid
            grid = (kyori // 0.5)
            # 周囲-5ボクセル～+5ボクセルの範囲を計算
            x_range = np.linspace(grid[0]-5, grid[0]+5, 11).astype(np.int32)
            y_range = np.linspace(grid[1]-5, grid[1]+5, 11).astype(np.int32)
            z_range = np.linspace(grid[2]-5, grid[2]+5, 11).astype(np.int32)
            # 各ボクセルに対する寄与値を計算
            for x in range(np.max(np.array([x_range.min(), 0])), np.min(np.array([x_range.max(), voxel.shape[1]])), 1):
                for y in range(np.max(np.array([y_range.min(), 0])), np.min(np.array([y_range.max(), voxel.shape[2]])), 1):
                    for z in range(np.max(np.array([z_range.min(), 0])), np.min(np.array([z_range.max(), voxel.shape[3]])), 1):
                        dd = np.array([Atom_x[i],Atom_y[i],Atom_z[i]])
                        voxel[atm[atype[i]], x, y, z] += 1.0 - math.exp(-pow(van[atype[i]] / np.linalg.norm(np.array([x/2,y/2,z/2]) + R_Grid - dd) , 12))
        
        # イオンを含む場合と含まない場合で返す配列の大きさを分岐
        if voxel[5:,:,:,:].sum()==0.:return voxel[:5,:,:,:]
        else: return voxel
    
    def define_hako(self, offset): 
        # 予測する範囲のボックスサイズを決定する。箱の中心×n + 削られる範囲(箱の大きさ-箱の中心)の大きさ
        center,split_size, arr = self.center, self.split_size, self.provoxel
        hakosize=np.zeros((3)).astype(np.int32)
        edge = split_size-center
        for xyz in range(3):
            size,cnt = 0,0
            # 予測領域（＝箱の中心×n）の最小の大きさを決定する
            # 3DRISMのボックスサイズから両端x/2だけ削った領域を最小の予測領域とする
            # デフォルトはx=24 ：　両端から6A削る
            while(arr.shape[xyz+1] - offset > size-edge):
                size+=center
            hakosize[xyz] = size
            
        # 予測用と3DRISMのボックスサイズを比較し、各辺における差を記録
        lost = (hakosize - np.array(list(arr.shape[1:]))).astype(np.int32)
        #print(lost)
        # 各辺における差を記録。複合時に使用
        self.lost = lost
        # 3辺について、3DRISMのボックスから削る大きさを求める。3DRISMより大きいときは0=削らない
        lost_ = -(np.minimum(lost, 0)//2)
        # 3DRISMのボックスから削る
        arr = arr[:, lost_[0]:arr.shape[1]-lost_[0], lost_[1]:arr.shape[2]-lost_[1], lost_[2]:arr.shape[3]-lost_[2]]
        # 差を更新して3DRISMよりどれだけ大きいか保存
        lost = (hakosize - np.array(list(arr.shape[1:]))).astype(np.int32)
        # 差に従って3DRISMのボックスに足す。渡された配列が水の場合1で埋め、タンパクの場合0で埋める
        #print(lost)
        if arr.shape[0] == 1:
            x_up = np.ones((arr.shape[0], lost[0]//2, arr.shape[2], arr.shape[3]))
            y_up = np.ones((arr.shape[0], hakosize[0], lost[1]//2, arr.shape[3]))
            z_up = np.ones((arr.shape[0], hakosize[0], hakosize[1], lost[2]//2))
        elif arr.shape[0] != 1:
            x_up = np.zeros((arr.shape[0], lost[0]//2, arr.shape[2], arr.shape[3]))
            y_up = np.zeros((arr.shape[0], hakosize[0], lost[1]//2, arr.shape[3]))
            z_up = np.zeros((arr.shape[0], hakosize[0], hakosize[1], lost[2]//2))
        hako = np.concatenate([z_up, np.concatenate([y_up, np.concatenate([x_up, arr, x_up], axis=1), y_up], axis=2), z_up], axis=3)
        #print(self.arr.shape, self.lost)
        self.hakosize = hako.shape
        return hako
    
    def spliter(self, hako): 
        # 渡された箱を分割して返す
        mi=[]
        center, size = self.center, self.split_size
        arr = hako
        # 端の大きさ
        cutting = (size-center)//2
        # x,y,z方向に何回分割できるか
        self.x_range, self.y_range, self.z_range = (arr.shape[1]-cutting*2)//center, (arr.shape[2]-cutting*2)//center, (arr.shape[3]-cutting*2)//center
        #print(x_range, y_range, z_range)
        # 分割
        for x in range(self.x_range):
            for y in range(self.y_range):
                for z in range(self.z_range):
                    #print("{},{},{} = {:3} : {:3}, {:3} : {:3}, {:3} : {:3}".format(x,y,z,x*16+16, x*16+48-16, y*16+16, y*16+48-16, z*16+16, z*16+48-16))
                    mi.append(arr[:, x*center : x*center+size, y*center:y*center+size, z*center:z*center+size])
        t = np.stack(mi).transpose(0,2,3,4,1).astype(np.float32)
        #print(self.x_range, self.y_range, self.z_range)
        return t
    
    def merger(self, t):
        # 分割した箱を結合して返す
        center, size = self.center, self.split_size
        x_range, y_range, z_range = self.x_range, self.y_range, self.z_range
        mii=[]
        # 箱の中心範囲のインデックスを計算 (ex: 48, 16 => 16:32) 
        cut_from, cut_to = (size-center)//2, size-(size-center)//2
        for i in range(t.shape[0]):
            mii.append(t[i, cut_from:cut_to, cut_from:cut_to, cut_from:cut_to, :])
        tt = np.stack(mii)
        bb = []
        for i in range(tt.shape[0]//z_range):
            #print("{}:{}".format(i*6, i*6+6))
            aa = [tt[i] for i in range(i*z_range, i*z_range+z_range)]
            bb.append(np.concatenate(aa,axis=2))
        tt = np.stack(bb)
        bb = []
        for i in range(tt.shape[0]//y_range):
            aa = [tt[i] for i in range(i*y_range, i*y_range+y_range)]
            bb.append(np.concatenate(aa,axis=1))
        tt = np.stack(bb)
        bb = []
        aa = [tt[i] for i in range(0, x_range)]
        bb.append(np.concatenate(aa,axis=0))
        tt = np.stack(bb)
        #print(tt.shape)
        
        edge = (size-center)//2
        
        # 局所予測のとき
        if self.pred_area_center[0] != None:
            return tt[0,:,:,:,0]
        
        pred_hako = np.ones((1, self.hakosize[1], self.hakosize[2], self.hakosize[3], 1))
        pred_hako[0, edge:self.hakosize[1]- edge, edge:self.hakosize[2] - edge, edge:self.hakosize[3] - edge, 0] = tt[0,:,:,:,0]
        #print(pred_hako.shape)
        lost_ = np.maximum(np.array(self.lost), 0)//2
        #print(lost_)
        pred_hako = pred_hako[:, lost_[0]:pred_hako.shape[1]-lost_[0], lost_[1]:pred_hako.shape[2]-lost_[1], lost_[2]:pred_hako.shape[3]-lost_[2], :]
        lost = (self.provoxel[0,:,:,:].shape - np.array(pred_hako[0,:,:,:,0].shape)).astype(np.int32)
        #print(pred_hako.shape, lost)
        x_up = np.ones((1, lost[0]//2, pred_hako.shape[2], pred_hako.shape[3], 1))
        y_up = np.ones((1, self.provoxel.shape[1], lost[1]//2, pred_hako.shape[3], 1))
        z_up = np.ones((1, self.provoxel.shape[1], self.provoxel.shape[2], lost[2]//2, 1))
        pred_hako = np.concatenate([z_up, np.concatenate([y_up, np.concatenate([x_up, pred_hako, x_up], axis=1), y_up], axis=2), z_up], axis=3)
    
        return pred_hako
        
    def predicter_normal(self, model_dir):
        
        fcn = tf.keras.models.load_model(model_dir, custom_objects={'lmse': lmse})
        hako = self.define_hako(offset=0)
        bara = self.spliter(hako)
        pred_index = np.array([i for i in range(bara.shape[0]) if bara[i,:,:,:,:].max() != 0]) 
        pred = fcn.predict(bara[pred_index])
        bara[:,:,:,:,:] = 1.
        bara[pred_index,:,:,:,:] = pred
        self.g_pred = self.merger(bara)[0,:,:,:,0]
        
    
    def predicter_local(self, model_dir):
        # 局所予測領域を決定する
        # 局所中心地の座標（ボクセル）
        px, py, pz = 2*(self.pred_area_center - self.R_Grid).astype(np.int32)
        if np.min([px,py,pz]) < 0 :
            print("ERROR : Invalid coordinate of center. Too far from the protein position.")
            return None
        #print(px,py,pz)
        # 局所領域の幅（ボクセル）
        rangE = int(self.pred_area_range*2)
        hakosize=np.zeros((3)).astype(np.int32)
        # 削られる幅
        edge = (self.split_size - self.center)
        for xyz in range(3):
            size = 0
            while(rangE*2  > size - edge):
                size+=self.center
            hakosize[:]=size
        #print(hakosize)
        local_area = np.zeros((3,2)) # ボクセル化に使う原子を決定する範囲　これ以内の原子を使う
        local_center_angstrom = self.R_Grid + np.array([px,py,pz])/2
        self.R_Grid = local_center_angstrom - hakosize/4 # 座標基準値の更新　多めに確保した予測領域分
            
        for xyz in range(3):
            local_area[xyz, 0] = local_center_angstrom[xyz]-hakosize[xyz]/2 -15
            local_area[xyz, 1] = local_center_angstrom[xyz]+hakosize[xyz]/2 +15
                
        # 局所予測領域のみのボクセル化    
        self.provoxel = self.proVoxelizer_Ion(self.atype.astype(np.int32), self.Atom_x, self.Atom_y, self.Atom_z, \
                                     self.R_Grid, np.array([hakosize[0], hakosize[1], hakosize[2]]).astype(np.int64), local_area)
            
        # 予測
        fcn = tf.keras.models.load_model(model_dir, custom_objects={'lmse': lmse})
        bara = self.spliter(self.provoxel)
        pred_index = np.array([i for i in range(bara.shape[0]) if bara[i,:,:,:,:].max() != 0])
        pred = fcn.predict(bara[pred_index])
        bara[:,:,:,:,:] = 1.
        bara[pred_index,:,:,:,:] = pred
        self.g_pred = self.merger(bara)                
                
        # ほしい領域の座標を切り取る
        cx, cy, cz = self.g_pred.shape[0]//2, self.g_pred.shape[1]//2, self.g_pred.shape[2]//2
        self.g_pred = self.g_pred[cx-rangE:cx+rangE,cy-rangE:cy+rangE,cz-rangE:cz+rangE]
        self.g_pred_for_compare = self.g_pred
        
        self.R_Grid = local_center_angstrom - self.pred_area_range # 座標基準値の更新 ほしい領域
        
        # 3DRISMがあれば比較用の正解g配列を作っておく
        if self.analysis_dir != None:
            # out of range を防ぐ
            fromx,fromy,fromz = np.max([px-rangE, 0]), np.max([py-rangE, 0]), np.max([pz-rangE, 0])
            tox,toy,toz = np.min([px+rangE, self.g_true.shape[0]]), np.min([py+rangE, self.g_true.shape[1]]), np.min([pz+rangE, self.g_true.shape[2]])
            #　スライス
            self.g_true_local = self.g_true[fromx:tox,fromy:toy,fromz:toz]
            # 予測配列と切り取った正解gの形が一致しないとき
            if self.g_true_local.shape != self.g_pred.shape :
                # out of range をした際の削られるボクセルを保存しとく
                leak = np.zeros((3,2))
                leak[0,0],leak[1,0],leak[2,0] = px-rangE, py-rangE, pz-rangE
                leak[:,0] = -np.minimum(leak[:,0],0)
                leak[0,1],leak[1,1],leak[2,1] = px+rangE, py+rangE, pz+rangE
                leak[:,1] = np.maximum(leak[:,1], \
                                                np.array([self.g_true.shape[0],self.g_true.shape[1],self.g_true.shape[2]])) \
                                                -np.array([self.g_true.shape[0],self.g_true.shape[1],self.g_true.shape[2]])
                leak = leak.astype(np.int32)
                # 正解gの削られる量に合わせて予測gも削って別の配列に保存
                self.g_pred_for_compare = self.g_pred[leak[0,0]:self.g_pred.shape[0]-leak[0,1],\
                                                      leak[1,0]:self.g_pred.shape[1]-leak[1,1],\
                                                      leak[2,0]:self.g_pred.shape[2]-leak[2,1]]
        
        
    # Converter; from npy to xyzv 
    def to_xyzv(self, data, save_dir):

        with open(save_dir, mode="w") as f:
            for z in range(data.shape[2]):
                for y in range(data.shape[1]):
                    for x in range(data.shape[0]):
                        moji_x =  " {:.8E} ".format(self.R_Grid[0] + x*0.5).replace("E+","E+0").replace("E-","E-0").replace(" -","-")
                        moji_y =  " {:.8E} ".format(self.R_Grid[1] + y*0.5).replace("E+","E+0").replace("E-","E-0").replace(" -","-") 
                        moji_z =  " {:.8E} ".format(self.R_Grid[2] + z*0.5).replace("E+","E+0").replace("E-","E-0").replace(" -","-")
                        moji_g =  " {:.8E} ".format(data[x, y, z]).replace("E+","E+0").replace("E-","E-0").replace(" -","-")
                        if moji_g[moji_g.rfind("E")+5] != ' ' :
                            moji_g = moji_g.replace("E+0", "E+").replace("E-0", "E-")
                        moji = moji_x + moji_y + moji_z + moji_g[:-1]+"\n"
                        f.write(moji)
                    
                    
    # Converter; from npy to dx 
    def to_dx(self, data, save_dir):

        import re
        nx, ny, nz = data.shape[0], data.shape[1], data.shape[2]

        with open(save_dir, "w") as f:
            f.write("object 1 class gridpositions counts{:8d}{:8d}{:8d}\n".format(nx, ny, nz))
            f.write("origin {:15.8f}{:15.8f}{:15.8f}\n".format(self.R_Grid[0], self.R_Grid[1], self.R_Grid[2]))
            f.write("delta       0.50000000 0 0\ndelta  0      0.50000000 0\ndelta  0 0      0.50000000\n")
            f.write("object 2 class gridconnections counts{:9d}{:9d}{:9d}\n".format(nx, ny, nz))
            f.write("object 3 class array type double rank 0 items {:27d} data follows\n".format(nx * ny * nz))
            idx=0
            amari = (nx * ny * nz) % 3
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        idx+=1
                        moji = "    0." + "{:.4E}".format(data[x, y, z]*10.).replace(".","").replace("E+","E+0").replace("E-","E-0")
                        try : 
                            if moji.find("0.-") != -1 : moji = re.sub("0.-(....).E", "-0.\\1E", moji)
                            if moji[moji.rfind("E")+5] != None : moji = moji.replace("+0","+").replace("-0","-")
                        except : pass
                        if idx % 3 == 0 :
                            f.write(moji+"\n")
                        else : 
                            f.write(moji)
                        if nx * ny * nz == idx and amari!=0:
                            f.write("\n")
            f.write("object \"Untitled\" class field\n")
