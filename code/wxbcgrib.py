# -*- coding: utf-8 -*-
"""
本スクリプトはWXBC人材育成WGの有志がテクノロジー研修「メッシュ気象データ分析チャ
レンジ！」のために作成したものですが、今般、オープンソース「MITライセンス」とし
て公開します。MITライセンスのルールに則りご利用いただき、ブラッシュアップしてい
ただけることを祈念しています。


The MIT License
Copyright 2021 気象ビジネス推進コンソーシアム

　以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウ
ェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱 うことを無償
で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、
サブライセンス、および/または販売する権利、および ソフトウェアを提供する相手に同
じことを許可する権利も無制限に含まれます。
　ただし、上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重
要な部分に記載するものとします。

　ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証も
なく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非
侵害についての保証も含みますが、それに限定されるものではありません。 作者または
著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因また
は関連し、あるいはソフトウェアの使用また はその他の扱いによって生じる一切の請求、
損害、その他の義務について何らの 責任も負わないものとします。
"""


import numpy as np  # 多次元配列計算用モジュール
import netCDF4 as nc  #NetCDFファイル操作のためのモジュール
import os            # ファイルやディレクトリの操作をするモジュール
import subprocess     # コマンドラインプログラムを実行させるモジュール
from datetime import datetime as dt   # 日付と時刻を示すdatetimeオブジェクトをを利用するためのモジュール
from datetime import timedelta as td   # 時間間隔を表現するtimedeltaオブジェクトを利用するためのモジュール
import scipy.interpolate as ip   # 補間を行うモジュール
import glob                      # ファイルやディレククトリの探索を行うモジュール
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt      # 描画を行うモジュール
import matplotlib.cm as cm          # 描画を行うモジュール

#from mpl_toolkits.basemap import Basemap # 図中に海岸線を埋め込むモジュール
#メソッドyx()実行時に「KeyError:'PROJ_LIB'」というエラーが出る場合は、以下のコメントを外し、あなたのユーザー名を指定場所に書いてください。
#os.environ['PROJ_LIB'] = r'C:/Users/（ユーザ名）/anaconda3/Library/share'

import cartopy.crs as ccrs  # 図中に海岸線を埋め込むモジュール


# WGRIB2の実行ファイルのを置くディレクトリを以下で指定
wgrib2 = "wgrib2"

# wgrib2 = "C:/wgrib2/wgrib2"   # Windowsの場合
#wgrib2 = "~/work/grib2/wgrib2/"    # Macの場合


def nc_path(grpath):
    """GRIBファイルへのパスから、NetCDFファイルへのパスを作る関数"""
    ncdir = "./nc"
    if not os.path.isdir(ncdir):    #NetCDFファイル置き場がなければ作る
        os.makedirs(ncdir)
    ncpath = os.path.join(ncdir, os.path.basename(grpath))+".nc"
    return ncpath


def nc_open(grpath):
    """GRIBファイルの中身をNetCDFファイルに変換してそれをオープンする関数"""
    ncpath = nc_path(grpath)
    if not os.path.isfile(ncpath):    #NetCDFに変換済みか確認
        rc = subprocess.run(f'{wgrib2} -netcdf {ncpath} {grpath}', 
                            shell=True, 
                            check=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            universal_newlines=True)
#        for line in rc.stdout.splitlines():
#            print('>>> ' + line)
    ds = nc.Dataset(ncpath)
    return ds


class Msg:
    """
    gribメッセージのようなもの　以下の属性を持つ
    name:名前(文字列)
    units:単位(文字列)
    _FillValue:無効値(浮動小数)
    time:時刻(datetimeオブジェクト)
    fcst:予報時間(整数(h))
    lats:緯度(1次元浮動小数)
    lons:経度(1次元浮動小数)
    data:データ本体(2次元浮動少数)
    """
    def __init__(self):  #Data() で作られると空のオブジェクトを返す
        self.tini = None
        self.name = None
        self.unit = None
        self._FillValue = None
        self.fcst = None
        self.data = None
        self.lat = None
        self.lon = None

    def __str__(self):
        return "".join([
                   f"tini={self.tini}:{self.name}:{self.unit}:",
                   f"{self.fcst} hour fcst:",
                   f"Shape={self.lat.shape[0]}x{self.lon.shape[0]}:",
                   f"FillValue={self._FillValue}"])

    def subset(self, lalomima):
        """
        self.dataに格納されているデータを、特定の領域部分だけにトリミングするメソッド
        緯度下限(南端)、緯度上限(北端)、経度下限(西端)、軽度上限(東端)の４つの
        数値をこの順でリストにして与える
        """
        ilami, ilama = self.idxsubset(self.lat, [lalomima[0],lalomima[1]])
        self.lat = self.lat[ilami:ilama+1]
        ilomi, iloma = self.idxsubset(self.lon, [lalomima[2],lalomima[3]])
        self.lon = self.lon[ilomi:iloma+1]
        self.data = self.data[ilami:ilama+1,ilomi:iloma+1]

    @staticmethod
    def idxsubset(gri, minmax):
        """端とグリッドが一致すればそのグリッド、一致しなかったら外側のグリッドの番号を返す"""
        imin = max(np.where(gri <= min(minmax))[0])
        imax = min(np.where(gri >= max(minmax))[0])
        return imin,imax


def msg_from_a_grnc(path,label, lalomima=None):
    """
    NetCDFファイルに変換したGRIBデータから'label'のデータを取り出して Msgオブジェクトのリストとして
    取り出す関数 (DS_from_grncで使う)
    キーワード引数latlonminmaxに緯度の下限、緯度の上限、軽度の下限、経度の上限をリストで与
    とその範囲でトリミングするしたメッセージを返す
    """
    tiniUTC = dt.strptime(os.path.basename(path).split("_")[4], # ファイル名から初期値時刻
                       "%Y%m%d%H%M%S")
    ncds = nc_open(path)
    nclabels = list(set(ncds.variables.keys()) - {"time","latitude","longitude"})
    msgs = []
    if label in nclabels:
        for i in range(len(ncds["time"])):
            m = Msg()
            m.tini = tiniUTC + td(hours=9)  # ファイル名やGRIBメッセージはUTCだがNetCDFファイルのデータセットはJST
            m.name = ncds[label].long_name
            m.unit = ncds[label].units
            m._FillValue = ncds[label]._FillValue
            m.fcst = (dt.fromtimestamp((int)(ncds["time"][i]))
                      - m.tini).total_seconds() / 3600.
            m.data = ncds[label][i,:,:]
            m.lat = ncds["latitude"][:]
            m.lon = ncds["longitude"][:]
            if lalomima != None:
                m.subset(lalomima)
            msgs.append(m)
    else:
        print("!! ラベルは以下から選択してください:")
        for lbl in nclabels:
            print(" "*4,lbl)
        raise ValueError
    nc.Dataset.close(ncds)
    return msgs


def DS_from_grnc(paths,label, lalomima=None):
    """GRIBファイルのデータををNetCDFファイル経由で読み込んでDataSetオブジェクトにする関数"""
    msgs = []
    if not isinstance(paths,list):
        paths = glob.glob(paths)
    for path in paths:
        #print(path)
        msgs = msgs + msg_from_a_grnc(path,label, lalomima)
    msgs.sort(key = lambda x: (x.tini,x.fcst))
    DS = DataSet(msgs)
    return DS


def msg_from_ndarray(time,lat,lon,data,name,unit="",_FillValue=-99999,tini=None):
    """
    ndarrayから Msgオブジェクトを構成する関数
    引数は以下の通り
        time:時刻の並び(datetimeオブジェクト)
        lat :緯度の並び(浮動小数)
        lon :経度の並び(浮動小数)
        data:2次元データ(浮動少数)
        name:名前(文字列)
        unit:単位(文字列)
        _FillValue:無効値(浮動小数)
        tini:初期時刻(datetimeオブジェクト)
    """
    msgs = []
    try:
        tim = np.array(time)
        lat = np.array(lat, dtype=np.float64)
        lon = np.array(lon, dtype=np.float64)
        dat = np.array(data, dtype=np.float64)
    except:
        print("Could not convert all data")
        raise
    data_shape = dat.shape
    domain_shape = (tim.shape[0],lat.shape[0],lon.shape[0])
#    print(f"data shape: {data_shape}")
#    print(f"time/lats/lon shape: {domain_shape}")
    if data_shape != domain_shape:
        print("Shape mismatch")
        raise ValueError
    if tini == None:
        tini = tim[0]
    for i in range(len(tim)):
        m = Msg()
        m.tini = tini
        m.name = name
        m.unit = unit
        m._FillValue = _FillValue
        m.fcst = (tim[i] - tini).total_seconds() / 3600.
        m.data = data[i,:,:]
        m.lat = lat
        m.lon = lon
        msgs.append(m)
    return msgs


def DS_from_ndarray(time,lat,lon,data,name,unit="",_FillValue=-99999,tini=None):
    """ndarrayからDataSetオブジェクトを作る関数"""
    msgs = msg_from_ndarray(time,lat,lon,data,name,unit,_FillValue,tini)
    DS = DataSet(msgs)
    return DS

def DS_like(ds, data=None, name=None, unit=None, _FillValue=None):
    """
    既存のDataSetをコピーするメソッド
    データと単位、無効値は入れ替えることも多いのでオプション引数でうけられるようにしてある
    """
    if data is None:
        data = ds.data
    if name is None:
        name = ds.name
    if unit is None:
        unit = ds.unit
    if _FillValue is None:
        _FillValue = ds.FillValue
    msgs = msg_from_ndarray(ds.time,ds.lat,ds.lon,data,name,unit,_FillValue,None)
    DS = DataSet(msgs)
    return DS



class DataSet:
    """
    Msgオブジェクトを集成して作成したデータセット
    以下の属性を持つ （ Msgオブジェクトとは同一でない）
        time:時刻の並び(datetimeオブジェクト)
        lat :緯度の並び(浮動小数)
        lon :経度の並び(浮動小数)
        name:名前(文字列)
        units:単位(文字列)
        _FillValue:無効値(浮動小数)
        time:時刻(datetimeオブジェクト)
        fcst:予報時間(整数(h))
        data:データ本体(2次元浮動少数)

    Msgオブジェクトには時刻の参照時刻と予報時間が保存されているが、DataSetオブジェクトにはそれらは
    なく、(初期値であれ予報値であれ)そのデータが指し示す時刻を属性に持つので注意。

    """
    def __init__(self,msgs=None):
        self.time = None
        self.lat = None
        self.lon = None
        self.data = None
        self.name = None
        self.unit = None
        self._FillValue = None
        if msgs:
            self.from_msgs(msgs)


    def from_msgs(self, msgs):
        """
        Msgオブジェクトのリストを固めてDataSetオブジェクトにするメソッド
        参照時刻と予報時間から、時刻(self.time)を計算する。名称等他の属性は、リストの最初
        のメッセージから取得して与える。参照時刻、予報時間でリストをソートしてから結合する
        """
        msgs.sort(key = lambda x: (x.tini,x.fcst))
        self.time = np.array([msg.tini + td(0,msg.fcst*3600) for msg in msgs])
        self.lat = np.array(msgs[0].lat)
        self.lon = np.array(msgs[0].lon)
        self.name = msgs[0].name
        self.unit = msgs[0].unit
        self._FillValue = msgs[0]._FillValue
        self.data = np.array([msg.data for msg in msgs])


    def from_ndarray(self,time,lat,lon,data,name,unit="",_FillValue=-99999,tini=None):
        """
        numpy ndarray で与えられた時刻や緯度経度、データからDataSetオブジェクトを構成するメソッド
        """
        msgs = msg_from_ndarray(time,lat,lon,data,name,unit,_FillValue,tini)
        self._from_msgs(msgs)
 
 
    def to_msgs(self, tini=None):
        """
        DataSetオブジェクトをバラバラにしたMsgオブジェクトのリストを返すメソッド
        オプション引数tiniが与えられない場合は、最も若い時刻を参照時刻とみなして予報時間を
        作成する。tiniが与えられた場合は、これに基づいて予報時刻を計算する。
        """
        msgs = msg_from_ndarray(self.time,self.lat,self.lon,self.data,
                                self.name,self.unit,self._FillValue,tini)
        return msgs


    def __str__(self):
        d = self.data.reshape(-1)
        nofi = len(d[d==self._FillValue])
        if len(d):
            dmin = np.min(d)
            davg = np.average(d)
            dmax = np.max(d)
        else:
            dmin = "---"
            davg = "---"
            dmax = "---"
        return "\n".join(
            [f"Name: {self.name} Unit: {self.unit}",
             f"Time: {self.time[0]} - {self.time[-1]}",
             f"Shape: {self.time.shape[0]} x {self.lat.shape[0]} x {self.lon.shape[0]}",
             f"Min: {dmin} Avg: {davg} Max: {dmax}",
             f"FillVal: {self._FillValue} (# of FillVal: {nofi})"])


    def lfm(self):
        """
        LFM-GPVと同じグリッドに空間補間された新しいDataSetオブジェクトを作るメソッド
        newds = ds.lfm() のように用いる
        """
        lats = np.linspace(22.4,47.6,1261)
#        lats = np.linspace(47.6,22.4,1261)
        lons = np.linspace(120,150,1201)
        return self.yx_interpolate(lats,lons)


    def tap(self):
        """
        推計気象分布と同じグリッド(3次メッシュ)に空間補間された新しいDataSetオブジェクトを作るメソッド
        newds = ds.tap() のように用いる
        """
        olat = 1/240.0 # 1/2 * (28/33600)
        olon = 1/160.0 # 1/2 * (32/2560)
        lats = np.linspace(20+olat,48-olat,3360)
        lons = np.linspace(118+olon,150-olon,2560)
        return self.yx_interpolate(lats,lons)


    def yx_interpolate(self,newlat,newlon):
        """
        引数で与えられた緯度経度のグリッドに空間補間された新しいDataSetオブジェクトを作るメソッド
        newds = ds.yx_interpolate(newlat,newlon) のように用いる
        """
#       外挿もあり得るのでこれらは使わない
#        if max(newlat) > max(self.lat) or min(newlat) < min(self.lat):
#            print("!! 指定した緯度がデータセットの範囲外です。")
#            raise ValueError
#        if max(newlon) > max(self.lon) or min(newlon) < min(self.lon):
#            print("!! 指定した経度がデータセットの範囲外です。")
#            raise ValueError
        newdata = np.zeros((len(self.time),len(newlat),len(newlon)),dtype=np.float32)
        for i in range(self.data.shape[0]):
            yx = self.data[i,:,:]
            f = ip.interp2d(self.lon,self.lat,yx)
            newdata[i,:,:] = f(newlon,newlat)
        newDS = DS_from_ndarray(self.time,newlat,newlon,newdata,self.name,
                                unit=self.unit,_FillValue=self._FillValue)
        return newDS


    def t_interpolate(self,newtime):
        """
        引数で与えられた時間間隔に空間補間された新しいDataSetオブジェクトを作るメソッド
        newds = ds.yx_interpolate(newlat,newlon) のように用いる
        """
        if max(newtime) > max(self.time) or min(newtime) < min(self.time):
            print("!! 指定した時刻がデータセットの範囲外です。")
            raise ValueError
        if newtime == self.time[0]:
            return self
        else:
            newdata = np.zeros((len(newtime),len(self.lat),len(self.lon)),dtype=np.float32)
            ts = [oo.timestamp() for oo in self.time]
            newts = [oo.timestamp() for oo in newtime]
            for i in range(self.data.shape[1]):
                for j in range(self.data.shape[2]):
                    vs = self.data[:,i,j]
                    f = ip.interp1d(ts,vs)
                    newdata[:,i,j] = f(newts)
                    newDS = DS_from_ndarray(newtime,self.lat,self.lon,newdata,self.name,
                                            unit=self.unit,_FillValue=self._FillValue)
            return newDS


    def yx(self,time,resolution="i",fig=False,prefix="yx",cmapstr=None, minmax=None):
        """
        引数で指定した時刻の気象値の分布を2次元配列で出力するメソッド。分布を画像として出力す
        ることもできる。
        書式：
            DS.yx(time,fig=False)　　または、　ret = DS.yx(time,fig=False)
        引数：
            time：必須。取得したい時刻
                Python datetimeオブジェクトで指定する。データセットに該当する時刻が無い場合は、
                前後の時刻から単純内挿する。この際、2段のforループで全ての地点グリッドに対して
                時間内挿を行うので、グリッド数が多いと処理に時間がかかる。
            fig：省略可。Trueを与えると、分布図をpngファイルで出力する。デフォルトではFalseが与えられる(すなわち図は作成されない)。
            resolution:省略可。fig=Trueのとき有効。描画する海岸線の精細度を
                'c'(粗い),’l’,'i','h','f'(細かい)で指定。精細にすると時間がとてもかかる。
            prefix：省略可。fig=Trueのとき有効。出力されるファイル名は、デフォルトで
                「Tyyyy.mm.dd.png」となるが、prefixに文字列を指定すると、その文字列がデフォ
                ルトファイル名のさらに前に付加される。ここで、yyyy、mm、ddは、順に、年、月、日を
                示す数字である。なお、ここにパスを書き込めば指定するディレクトリにファイルを配置できる。

            cmapstr：省略可。値と色との関連付けに付けられた呼称(カラーマップ)を字列で与える。
                省略された場合は'RdYlGn_r'が与えられる。
            minmax：省略可。配色の最小値、最大値を[min,max]で与える。
        戻り値：
            省略可。指定した時刻分布データを数値として取り出したいときは戻り値で受ける。
        カラーマップについて：
            カラーマップには名称があるのでこれを文字列で("で囲んで)指定する。
            例)　レインボーカラー:rainbow、黄色-オレンジ-赤の順で変化:YlOrRdなど
            色の順序をを反転させたい場合は、rainbow_rのよう名称の後ろに"_r"を付加する。
            詳細は下記URLを参照。
            http://matplotlib.org/examples/color/colormaps_reference.html
        """
        path = f"{prefix}T{dt.strftime(time,'%Y.%m%d.%H%M')}.png"
        it = np.where(self.time == time)
        itime = it[0]
        if len(itime) == 0:
            DS = self.t_interpolate(np.array([time]))
            yx = DS.data[0,:,:]
        else:
            yx = self.data[itime[0],:,:]
        if fig:
            myx = np.ma.masked_values(yx, self._FillValue)
            tate = 6      #図の台紙の全体的な大きさを指定します。
            yoko = tate * (np.max(self.lon)-np.min(self.lon))/(np.max(self.lat)-np.min(self.lat)) + 2
            plt.figure(figsize=(yoko,tate)) # プロット領域の作成（matplotlib）

            ax = plt.subplot(1,1,1, facecolor='0.8', 
                             projection=ccrs.PlateCarree())
            ax.set_extent([self.lon[0],self.lon[-1],self.lat[0],self.lat[-1]], ccrs.PlateCarree())
            ax.coastlines(lw=0.5)
            gl = ax.gridlines(draw_labels=True,lw=5)
            gl.top_labels = False
            gl.right_labels = False

            if cmapstr is None:
                cmapstr = 'RdYlGn_r'
            cmap = eval("cm."+cmapstr)
            if minmax is None:
                minmax = [None,None]
            cf = ax.pcolormesh(self.lon,self.lat,myx, 
                                vmin=minmax[0], vmax=minmax[1], 
                                cmap=cmap, shading='auto')

            plt.colorbar(cf)
            plt.title(f"{self.name} ({self.unit}) {time}")
            plt.savefig(path, format='png', dpi=300)
            plt.show()
            plt.close()
        return yx

    def ts(self,lat,lon,fig=False, csv=False, prefix="ts"):
        """
        引数で指定した緯度経度の気象値の時系列を1次元配列で出力するメソッド。時系列グラフを
        画像として出力することも可能。
        書式：
            DS.ts(time,fig=False)　または、ret = DS.ts(time,fig=False)
        引数：
        lat：必須。取得したい地点の緯度。十進数で指定する。
        lon：必須。取得したい地点の経度。十進数で指定する。
        fig：省略可。Trueを与えると、時系列折れ線グラフをpngファイルで出力する。
            デフォルトではFalseが与えられる(すなわち図は作成されない)。
        csv：省略可。Trueを与えると、データをテキストファイルに出力する。デフォルト
            ではFalseが与えられる(すなわちファイルは出力されない。
        prefix：省略可。fig=Trueのとき有効。出力される画像ファイル名は、
            デフォルトで「tsNlat-Elon.png」(グラフの場合)、「tsNlat-Elon.csv」となるが、
            prefixに文字列を指定すると、その文字列がデフォルトファイル名のさらに前に付加される。
            ここでlat、lonはそれぞれ緯度と経度を示す数字である。
        戻り値：
            省略可。指定した時系列データを数値として取り出したいときは戻り値で受ける。
        """
        DS = self.yx_interpolate(np.array([lat]),np.array([lon]))
        xs = DS.time
        ys = DS.data.reshape(-1)
        if fig:
            path = f"{prefix}N{lat}-E{lon}.png"
            plt.figure(figsize=(12, 4)) # figureの縦横の大きさ
            plt.subplot(1,1,1)
            plt.plot(xs,ys)
            plt.title(f"{self.name} ({self.unit}) lat:{lat} lon:{lon}")
            plt.savefig(path, format='png', dpi=300)
            plt.show()
            plt.close()
        if csv:
            path = f"{prefix}N{lat}-E{lon}.csv"
            with open(path, "w") as f:
                print('time'+", "+f'{self.name}', file=f)
                for i in range(len(xs)):
                    print(dt.strftime(xs[i],'%Y/%m/%d %H:%M:%S')+", "
                          '{:.2f}'.format(ys[i]), file=f)
        return ys


    def ql(self,itime=0,cmapstr=None, minmax=None):
        """
        DataSetオブジェクトの最初の時刻の様子を手っ取り早く表示させるメソッド。DataSetオブジェクト
        のサイズ等も表示する。
        引数に整数を与えるとその整数に対応する時刻の分布図を示す。最初は0、最後は-1。
        cmapstr：省略可。値と色との関連付けに付けられた呼称文字列を与える。省略された
            場合は'RdYlGn_r'が与えられる。
        minmax：省略可。配色の最小値を要素2のリストで与える。
        """
        dat = np.ma.masked_values(self.data[itime,:,:], self._FillValue)
        tate = 6      #図の台紙の全体的な大きさを指定します。
        yoko = tate * (np.max(self.lon)-np.min(self.lon))/(np.max(self.lat)-np.min(self.lat)) + 2
        plt.figure(figsize=(yoko,tate)) # プロット領域の作成（matplotlib）
        ax = plt.subplot(1,1,1, facecolor='0.8')
        if cmapstr is None:
            cmapstr = 'RdYlGn_r'
        cmap = eval("cm."+cmapstr)
        if minmax is None:
            minmax = [None,None]
        cf = ax.imshow(dat, cmap=cmap, origin='lower', aspect='equal',
                          vmin=minmax[0], vmax=minmax[1])
        plt.colorbar(cf)
        plt.title(f"{self.name} ({self.unit}) {self.time[itime]}")
        plt.show()
        plt.close()

        d1 = np.array(dat).reshape(-1)
        nofi = len(d1[d1==self._FillValue])
        print("\n".join(
            [f"Name: {self.name} Unit: {self.unit}",
             f"Time: {self.time[itime]}",
             f"lat: {self.lat[0]} - {self.lat[-1]}  ({self.lat[1]-self.lat[0]})",
             f"lon: {self.lon[0]} - {self.lon[-1]}  ({self.lon[1]-self.lon[0]})",
             f"Shape: {self.time.shape[0]} x {self.lat.shape[0]} x {self.lon.shape[0]}",
             f"FillVal: {self._FillValue} (# of FillVal: {nofi})"]))
