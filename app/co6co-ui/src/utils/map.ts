import { Point } from '@/constants';

// 定义一些常量
const x_PI = (3.14159265358979324 * 3000.0) / 180.0;
const PI = 3.1415926535897932384626;
const a = 6378245.0;
const ee = 0.00669342162296594323;
/**
 * 百度坐标系 (BD-09) 与 火星坐标系 (GCJ-02) 的转换
 * 即 百度 转 高德
 * @param point
 * @returns Point
 */
export const bd09togcj02 = (point: Point): Point => {
	let x = point.lng - 0.0065;
	let y = point.lat - 0.006;
	let z = Math.sqrt(x * x + y * y) - 0.00002 * Math.sin(y * x_PI);
	let theta = Math.atan2(y, x) - 0.000003 * Math.cos(x * x_PI);
	let gg_lng = z * Math.cos(theta);
	let gg_lat = z * Math.sin(theta);
	return { lng: gg_lng, lat: gg_lat };
};

/**
 * 火星坐标系 (GCJ-02) 与百度坐标系 (BD-09) 的转换
 * 即  高德 转 百度
 * @param point
 * @returns Point
 */
export const gcj02tobd09 = (point: Point): Point => {
	let lat = point.lat;
	let lng = point.lng;
	let z = Math.sqrt(lng * lng + lat * lat) + 0.00002 * Math.sin(lat * x_PI);
	let theta = Math.atan2(lat, lng) + 0.000003 * Math.cos(lng * x_PI);
	let bd_lng = z * Math.cos(theta) + 0.0065;
	let bd_lat = z * Math.sin(theta) + 0.006;
	return { lng: bd_lng, lat: bd_lat };
};

/**
 * WGS-84 转 GCJ-02
 * @param point
 * @returns Point
 */
export const wgs84togcj02 = (point: Point): Point => {
	let lat = point.lat;
	let lng = point.lng;
	if (out_of_china(point.lng, point.lat)) {
		return point;
	} else {
		let dlat = transformlat(lng - 105.0, lat - 35.0);
		let dlng = transformlng(lng - 105.0, lat - 35.0);
		let radlat = (lat / 180.0) * PI;
		let magic = Math.sin(radlat);
		magic = 1 - ee * magic * magic;
		let sqrtmagic = Math.sqrt(magic);
		dlat = (dlat * 180.0) / (((a * (1 - ee)) / (magic * sqrtmagic)) * PI);
		dlng = (dlng * 180.0) / ((a / sqrtmagic) * Math.cos(radlat) * PI);
		let mglat = lat + dlat;
		let mglng = lng + dlng;

		return { lng: mglng, lat: mglat };
	}
};

/**
 * GCJ-02 转换为 WGS-84
 * @param point
 * @returns Point
 */
export const gcj02towgs84 = (point: Point): Point => {
	if (out_of_china(point.lng, point.lat)) {
		return point;
	} else {
		let lat = point.lat;
		let lng = point.lng;
		let dlat = transformlat(point.lng - 105.0, point.lat - 35.0);
		let dlng = transformlng(point.lng - 105.0, point.lat - 35.0);
		let radlat = (lat / 180.0) * PI;
		let magic = Math.sin(radlat);
		magic = 1 - ee * magic * magic;
		let sqrtmagic = Math.sqrt(magic);
		dlat = (dlat * 180.0) / (((a * (1 - ee)) / (magic * sqrtmagic)) * PI);
		dlng = (dlng * 180.0) / ((a / sqrtmagic) * Math.cos(radlat) * PI);
		let mglat = lat + dlat;
		let mglng = lng + dlng;
		return { lng: lng * 2 - mglng, lat: lat * 2 - mglat };
	}
};

export const transformlat = (lng: number, lat: number) => {
	let ret =
		-100.0 +
		2.0 * lng +
		3.0 * lat +
		0.2 * lat * lat +
		0.1 * lng * lat +
		0.2 * Math.sqrt(Math.abs(lng));
	ret +=
		((20.0 * Math.sin(6.0 * lng * PI) + 20.0 * Math.sin(2.0 * lng * PI)) *
			2.0) /
		3.0;
	ret +=
		((20.0 * Math.sin(lat * PI) + 40.0 * Math.sin((lat / 3.0) * PI)) * 2.0) /
		3.0;
	ret +=
		((160.0 * Math.sin((lat / 12.0) * PI) + 320 * Math.sin((lat * PI) / 30.0)) *
			2.0) /
		3.0;
	return ret;
};

export const transformlng = (lng: number, lat: number) => {
	let ret =
		300.0 +
		lng +
		2.0 * lat +
		0.1 * lng * lng +
		0.1 * lng * lat +
		0.1 * Math.sqrt(Math.abs(lng));
	ret +=
		((20.0 * Math.sin(6.0 * lng * PI) + 20.0 * Math.sin(2.0 * lng * PI)) *
			2.0) /
		3.0;
	ret +=
		((20.0 * Math.sin(lng * PI) + 40.0 * Math.sin((lng / 3.0) * PI)) * 2.0) /
		3.0;
	ret +=
		((150.0 * Math.sin((lng / 12.0) * PI) +
			300.0 * Math.sin((lng / 30.0) * PI)) *
			2.0) /
		3.0;
	return ret;
};

/**
 * 判断是否在国内，不在国内则不做偏移
 * @param lng
 * @param lat
 * @returns {boolean}
 */
const out_of_china = (lng: number, lat: number): boolean => {
	// 纬度 3.86~53.55, 经度 73.66~135.05
	return !isInChina(lng, lat);
};
// 辅助函数
export function isInChina(lng: number, lat: number): boolean {
	return lng > 73.66 && lng < 135.05 && lat > 3.86 && lat < 53.55;
}
