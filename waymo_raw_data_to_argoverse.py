


import imageio
import itertools
import json
import math
import numpy as np
import os
from pathlib import Path
import pdb
from typing import Any, Dict, Union

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

"""
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto

See paper:
https://arxiv.org/pdf/1912.04838.pdf
"""

from scipy.spatial.transform import Rotation

CAMERA_NAMES = [
    'unknown', # 0, 'UNKNOWN',
    'ring_front_center', # 1, 'FRONT'
    'ring_front_left', # 2, 'FRONT_LEFT',
    'ring_front_right', # 3, 'FRONT_RIGHT',
    'ring_side_left', # 4, 'SIDE_LEFT',
    'ring_side_right', # 5, 'SIDE_RIGHT'
]

def check_mkdir(dirpath):
	""" """
	if not Path(dirpath).exists():
		os.makedirs(dirpath, exist_ok=True)

def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
	"""Save a Python dictionary to a JSON file.
	Args:
	json_fpath: Path to file to create.
	dictionary: Python dictionary to be serialized.
	"""
	with open(json_fpath, "w") as f:
		json.dump(dictionary, f)

def main():
	""" """
	TFRECORD_DIR = '/export/share/Datasets/MSegV12/w_o_d/VAL_TFRECORDS'
	log_ids = [
		'11450298750351730790_1431_750_1451_750',
		'11406166561185637285_1753_750_1773_750',
		'16979882728032305374_2719_000_2739_000',
		'12831741023324393102_2673_230_2693_230',
		'14486517341017504003_3406_349_3426_349',
		'12358364923781697038_2232_990_2252_990',
		'6183008573786657189_5414_000_5434_000',
		'3126522626440597519_806_440_826_440',
		'17152649515605309595_3440_000_3460_000',
		'10359308928573410754_720_000_740_000',
		'16751706457322889693_4475_240_4495_240',
		'14165166478774180053_1786_000_1806_000',
		'14081240615915270380_4399_000_4419_000',
		'1071392229495085036_1844_790_1864_790',
		'18305329035161925340_4466_730_4486_730',
		'13336883034283882790_7100_000_7120_000',
		'11356601648124485814_409_000_429_000',
		'1943605865180232897_680_000_700_000',
		'13178092897340078601_5118_604_5138_604',
		'15488266120477489949_3162_920_3182_920',
		'12306251798468767010_560_000_580_000',
		'4612525129938501780_340_000_360_000',
		'260994483494315994_2797_545_2817_545',
		'2105808889850693535_2295_720_2315_720',
		'18446264979321894359_3700_000_3720_000',
		'8331804655557290264_4351_740_4371_740',
		'4013125682946523088_3540_000_3560_000',
		'14931160836268555821_5778_870_5798_870',
		'14300007604205869133_1160_000_1180_000',
		'13573359675885893802_1985_970_2005_970',
		'8079607115087394458_1240_000_1260_000',
		'18331704533904883545_1560_000_1580_000',
		'12866817684252793621_480_000_500_000',
		'12374656037744638388_1412_711_1432_711',
		'7493781117404461396_2140_000_2160_000',
		'15224741240438106736_960_000_980_000',
		'11616035176233595745_3548_820_3568_820',
		'10837554759555844344_6525_000_6545_000',
		'10689101165701914459_2072_300_2092_300',
		'8845277173853189216_3828_530_3848_530',
		'6680764940003341232_2260_000_2280_000',
		'30779396576054160_1880_000_1900_000',
		'17791493328130181905_1480_000_1500_000',
		'9024872035982010942_2578_810_2598_810',
		'8302000153252334863_6020_000_6040_000',
		'4246537812751004276_1560_000_1580_000',
		'346889320598157350_798_187_818_187',
		'16229547658178627464_380_000_400_000',
		'13356997604177841771_3360_000_3380_000',
		'1105338229944737854_1280_000_1300_000',
		'4816728784073043251_5273_410_5293_410',
		'18024188333634186656_1566_600_1586_600',
		'14956919859981065721_1759_980_1779_980',
		'14244512075981557183_1226_840_1246_840',
		'13184115878756336167_1354_000_1374_000',
		'12102100359426069856_3931_470_3951_470',
		'2506799708748258165_6455_000_6475_000',
		'14739149465358076158_4740_000_4760_000',
		'2308204418431899833_3575_000_3595_000',
		'18333922070582247333_320_280_340_280',
		'17763730878219536361_3144_635_3164_635',
		'12657584952502228282_3940_000_3960_000',
		'11901761444769610243_556_000_576_000',
		'17135518413411879545_1480_000_1500_000',
		'16767575238225610271_5185_000_5205_000',
		'14127943473592757944_2068_000_2088_000',
		'6324079979569135086_2372_300_2392_300',
		'5847910688643719375_180_000_200_000',
		'447576862407975570_4360_000_4380_000',
		'3015436519694987712_1300_000_1320_000',
		'271338158136329280_2541_070_2561_070',
		'9164052963393400298_4692_970_4712_970',
		'7932945205197754811_780_000_800_000',
		'5183174891274719570_3464_030_3484_030',
		'2736377008667623133_2676_410_2696_410',
		'17626999143001784258_2760_000_2780_000',
		'15724298772299989727_5386_410_5406_410',
		'12134738431513647889_3118_000_3138_000',
		'5990032395956045002_6600_000_6620_000',
		'5832416115092350434_60_000_80_000',
		'4195774665746097799_7300_960_7320_960',
		'3915587593663172342_10_000_30_000',
		'17136314889476348164_979_560_999_560',
		'14687328292438466674_892_000_912_000',
		'4575389405178805994_4900_000_4920_000',
		'14663356589561275673_935_195_955_195',
		'13469905891836363794_4429_660_4449_660',
		'12820461091157089924_5202_916_5222_916',
		'11660186733224028707_420_000_440_000',
		'9243656068381062947_1297_428_1317_428',
		'9041488218266405018_6454_030_6474_030',
		'8679184381783013073_7740_000_7760_000',
		'6637600600814023975_2235_000_2255_000',
		'2367305900055174138_1881_827_1901_827',
		'14811410906788672189_373_113_393_113',
		'11048712972908676520_545_000_565_000',
		'7253952751374634065_1100_000_1120_000',
		'4423389401016162461_4235_900_4255_900',
		'4409585400955983988_3500_470_3520_470',
		'2834723872140855871_1615_000_1635_000',
		'17244566492658384963_2540_000_2560_000',
		'15096340672898807711_3765_000_3785_000',
		'15021599536622641101_556_150_576_150',
		'1331771191699435763_440_000_460_000',
		'8133434654699693993_1162_020_1182_020',
		'5574146396199253121_6759_360_6779_360',
		'366934253670232570_2229_530_2249_530',
		'3077229433993844199_1080_000_1100_000',
		'3039251927598134881_1240_610_1260_610',
		'15611747084548773814_3740_000_3760_000',
		'1405149198253600237_160_000_180_000',
		'1024360143612057520_3580_000_3600_000',
		'4759225533437988401_800_000_820_000',
		'2335854536382166371_2709_426_2729_426',
		'1505698981571943321_1186_773_1206_773',
		'12496433400137459534_120_000_140_000',
		'8398516118967750070_3958_000_3978_000',
		'8137195482049459160_3100_000_3120_000',
		'17860546506509760757_6040_000_6060_000',
		'16204463896543764114_5340_000_5360_000',
		'15948509588157321530_7187_290_7207_290',
		'10868756386479184868_3000_000_3020_000',
		'7988627150403732100_1487_540_1507_540',
		'5772016415301528777_1400_000_1420_000',
		'3577352947946244999_3980_000_4000_000',
		'17612470202990834368_2800_000_2820_000',
		'10335539493577748957_1372_870_1392_870',
		'933621182106051783_4160_000_4180_000',
		'89454214745557131_3160_000_3180_000',
		'5289247502039512990_2640_000_2660_000',
		'3651243243762122041_3920_000_3940_000',
		'16213317953898915772_1597_170_1617_170',
		'14383152291533557785_240_000_260_000',
		'14333744981238305769_5658_260_5678_260',
		'15959580576639476066_5087_580_5107_580',
		'14262448332225315249_1280_000_1300_000',
		'13415985003725220451_6163_000_6183_000',
		'12940710315541930162_2660_000_2680_000',
		'11387395026864348975_3820_000_3840_000',
		'5302885587058866068_320_000_340_000',
		'4690718861228194910_1980_000_2000_000',
		'1906113358876584689_1359_560_1379_560',
		'9114112687541091312_1100_000_1120_000',
		'2624187140172428292_73_000_93_000',
		'13694146168933185611_800_000_820_000',
		'9472420603764812147_850_000_870_000',
		'9443948810903981522_6538_870_6558_870',
		'902001779062034993_2880_000_2900_000',
		'7799643635310185714_680_000_700_000',
		'7650923902987369309_2380_000_2400_000',
		'7119831293178745002_1094_720_1114_720',
		'1464917900451858484_1960_000_1980_000',
		'967082162553397800_5102_900_5122_900',
		'8888517708810165484_1549_770_1569_770',
		'4764167778917495793_860_000_880_000',
		'10247954040621004675_2180_000_2200_000',
		'6074871217133456543_1000_000_1020_000',
		'11434627589960744626_4829_660_4849_660',
		'9231652062943496183_1740_000_1760_000',
		'4490196167747784364_616_569_636_569',
		'10448102132863604198_472_000_492_000',
		'6491418762940479413_6520_000_6540_000',
		'13299463771883949918_4240_000_4260_000',
		'10203656353524179475_7625_000_7645_000',
		'662188686397364823_3248_800_3268_800',
		'17962792089966876718_2210_933_2230_933',
		'8956556778987472864_3404_790_3424_790',
		'8907419590259234067_1960_000_1980_000',
		'8506432817378693815_4860_000_4880_000',
		'2551868399007287341_3100_000_3120_000',
		'7163140554846378423_2717_820_2737_820',
		'6707256092020422936_2352_392_2372_392',
		'6001094526418694294_4609_470_4629_470',
		'17344036177686610008_7852_160_7872_160',
		'15396462829361334065_4265_000_4285_000',
		'13941626351027979229_3363_930_3383_930',
		'15496233046893489569_4551_550_4571_550',
		'18252111882875503115_378_471_398_471',
		'17539775446039009812_440_000_460_000',
		'4426410228514970291_1620_000_1640_000',
		'7732779227944176527_2120_000_2140_000',
		'5373876050695013404_3817_170_3837_170',
		'14107757919671295130_3546_370_3566_370',
		'10289507859301986274_4200_000_4220_000',
		'17703234244970638241_220_000_240_000',
		'18045724074935084846_6615_900_6635_900',
		'9579041874842301407_1300_000_1320_000',
		'6161542573106757148_585_030_605_030',
		'2094681306939952000_2972_300_2992_300',
		'17694030326265859208_2340_000_2360_000',
		'15028688279822984888_1560_000_1580_000',
		'9265793588137545201_2981_960_3001_960',
		'17065833287841703_2980_000_3000_000',
		'5372281728627437618_2005_000_2025_000',
		'3731719923709458059_1540_000_1560_000',
		'14624061243736004421_1840_000_1860_000',
		'13982731384839979987_1680_000_1700_000',
		'191862526745161106_1400_000_1420_000',
		'1457696187335927618_595_027_615_027',
		'11037651371539287009_77_670_97_670',
		'4854173791890687260_2880_000_2900_000',
		'272435602399417322_2884_130_2904_130'
	]

	save_images = False
	save_poses = False

	img_count = 0
	for log_id in log_ids:
		print(log_id)
		tfrecord_name = f'segment-{log_id}_with_camera_labels.tfrecord'
		tf_fpath = f'{TFRECORD_DIR}/{tfrecord_name}'
		dataset = tf.data.TFRecordDataset(tf_fpath, compression_type='')
		
		log_calib_json = None

		for data in dataset:
			frame = open_dataset.Frame()
			frame.ParseFromString(bytearray(data.numpy()))
			# discovered_log_id = '967082162553397800_5102_900_5122_900'
			assert log_id == frame.context.name
			
			# Frame start time, which is the timestamp of the first top lidar spin
			# within this frame, in microseconds
			timestamp_ms = frame.timestamp_micros
			timestamp_ns = int(timestamp_ms * 1000) # to nanoseconds
			SE3_flattened = np.array(frame.pose.transform)
			city_SE3_egovehicle = SE3_flattened.reshape(4,4)
			if save_poses:
				dump_pose(city_SE3_egovehicle, timestamp_ns, log_id)

			
			calib_json = form_calibration_json(frame.context.camera_calibrations)
			if log_calib_json is None:
				log_calib_json = calib_json

				calib_json_fpath = f'pose_logs/{log_id}/vehicle_calibration_info.json'
				check_mkdir(str(Path(calib_json_fpath).parent))
				save_json_dict(calib_json_fpath, calib_json)

			else:
				# pdb.set_trace()
				assert calib_json == log_calib_json	

			# 5 images per frame
			for index, tf_cam_image in enumerate(frame.images):

				# 4x4 row major transform matrix that tranforms 
				# 3d points from one frame to another.
				SE3_flattened = np.array(tf_cam_image.pose.transform)
				city_SE3_egovehicle = SE3_flattened.reshape(4,4)

				# microseconds
				timestamp_ms =  tf_cam_image.pose_timestamp
				timestamp_ns = int(timestamp_ms * 1000) # to nanoseconds
				# tf_cam_image.shutter
				# tf_cam_image.camera_trigger_time
				# tf_cam_image.camera_readout_done_time
				if save_poses:
					dump_pose(city_SE3_egovehicle, timestamp_ns, log_id)

				if save_images:
					camera_name = CAMERA_NAMES[tf_cam_image.name]
					img = tf.image.decode_jpeg(tf_cam_image.image)
					img_save_fpath = f'logs/{log_id}/{camera_name}/{camera_name}_{timestamp_ns}.jpg'
					assert not Path(img_save_fpath).exists()
					check_mkdir(str(Path(img_save_fpath).parent))
					imageio.imwrite(img_save_fpath, img)
					img_count += 1
					if img_count % 100 == 0:
						print(f"\tSaved {img_count}'th image for log = {log_id}")

				
				# pose_save_fpath = f'logs/{log_id}/poses/city_SE3_egovehicle_{timestamp_ns}.json'
				# assert not Path(pose_save_fpath).exists()
				# save_json_dict(pose_save_fpath)


RING_IMAGE_SIZES = {
	# width x height
	'ring_front_center': (1920, 1280),
	'ring_front_left':  (1920, 1280),
	'ring_side_left': (1920, 886),
	'ring_side_right': (1920,886)
}

def form_calibration_json(calib_data):
	""" """
	calib_dict = {
		'camera_data_': []
	}
	for camera_calib in calib_data:

		cam_name = CAMERA_NAMES[camera_calib.name]
		city_SE3_egovehicle = np.array(camera_calib.extrinsic.transform).reshape(4,4)
		x,y,z = city_SE3_egovehicle[:3,3]
		R = city_SE3_egovehicle[:3,:3]
		assert np.allclose( city_SE3_egovehicle[3], np.array([0,0,0,1]) )
		q = rotmat2quat(R)
		qw, qx, qy, qz = q
		f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic

		cam_dict = {
			'key': 'image_raw_' + cam_name,
			'value': {
				'focal_length_x_px_': f_u,
				'focal_length_y_px_': f_v,
				'focal_center_x_px_': c_u,
				'focal_center_y_px_': c_v,
				'skew_': 0,
				'distortion_coefficients_': [0,0,0],
				'vehicle_SE3_camera_': {
					'rotation': {'coefficients': [qw, qx, qy, qz] },
					'translation': [x,y,z]
				}
			}
		}
		calib_dict['camera_data_'] += [cam_dict]

	return calib_dict


def dump_pose(city_SE3_egovehicle, timestamp, log_id):
	""" """
	x,y,z = city_SE3_egovehicle[:3,3]
	R = city_SE3_egovehicle[:3,:3]
	assert np.allclose( city_SE3_egovehicle[3], np.array([0,0,0,1]) )
	q = rotmat2quat(R)
	w, x, y, z = q
	pose_dict = {
		'rotation': [w, x, y, z],
		'translation': [x,y,z]
	}
	json_fpath = f'pose_logs/{log_id}/poses/city_SE3_egovehicle_{timestamp}.json'
	check_mkdir(str(Path(json_fpath).parent))
	save_json_dict(json_fpath, pose_dict)


def rotmat2quat(R: np.ndarray) -> np.ndarray:
	""" """
	q_scipy = Rotation.from_dcm(R).as_quat()
	x, y, z, w = q_scipy
	q_argo = w, x, y, z
	return q_argo


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit-length quaternion into a rotation matrix.
    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.
    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates
    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return Rotation.from_quat(q_scipy).as_dcm()


def test_cycle():
	""" """
	R = np.eye(3)
	q = rotmat2quat(R)
	R_cycle = quat2rotmat(q)
	


if __name__ == '__main__':
	main()
	# test_cycle()



