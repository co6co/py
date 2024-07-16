import { Point } from '@/constants';
export function gcj02ToWgs84(point: Point): Point {
	let dLat = transformLat(point.lng - 105.0, point.lat - 35.0);
	let dLng = transformLng(point.lng - 105.0, point.lat - 35.0);
	const radLat = (point.lat / 180) * Math.PI;
	let magic = Math.sin(radLat);
	magic = 1 - 0.00669342162296594323 * magic * magic;
	const sqrtMagic = Math.sqrt(magic);
	dLat =
		(dLat * 180.0) / ((point.lat * 0.00669342162296594323) / 180.0 / sqrtMagic);
	dLng =
		(dLng * 180.0) /
		((point.lng * 0.00669342162296594323) / 180.0 / (magic * sqrtMagic));
	return {
		lng: point.lng + dLng,
		lat: point.lat + dLat,
	};
}

export function wgs84ToGcj02(point: Point): Point {
	if (isInChina(point.lng, point.lat)) {
		const dLat = transformLat(point.lng - 105.0, point.lat - 35.0);
		const dLng = transformLng(point.lng - 105.0, point.lat - 35.0);
		return {
			lng: point.lng + dLng,
			lat: point.lat + dLat,
		};
	}
	return point;
}

export function bd09ToGcj02(point: Point): Point {
	let x = point.lng,
		y = point.lat;
	const z = Math.sqrt(x * x + y * y) + 0.00002 * Math.sin(y * Math.PI);
	const theta = Math.atan2(y, x) + 0.000003 * Math.cos(x * Math.PI);
	return {
		lng: z * Math.cos(theta) + 0.0065,
		lat: z * Math.sin(theta) + 0.006,
	};
}

export function gcj02ToBd09(point: Point): Point {
	let x = point.lng,
		y = point.lat;
	const z = Math.sqrt(x * x + y * y) + 0.00002 * Math.sin(y * Math.PI);
	const theta = Math.atan2(y, x) + 0.000003 * Math.cos(x * Math.PI);
	return {
		lng: z * Math.cos(theta) + 0.0065,
		lat: z * Math.sin(theta) + 0.006,
	};
}

export function wgs84ToBd09(point: Point): Point {
	return gcj02ToBd09(wgs84ToGcj02(point));
}

export function bd09ToWgs84(point: Point): Point {
	return gcj02ToWgs84(bd09ToGcj02(point));
}

// 辅助函数
export function isInChina(lng: number, lat: number): boolean {
	return lng > 73.66 && lng < 135.05 && lat > 3.86 && lat < 53.55;
}

function transformLat(x: number, y: number): number {
	return (
		-100.0 +
		2.0 * x +
		3.0 * y +
		0.2 * y * y +
		0.1 * x * y +
		0.2 * Math.sqrt(Math.abs(x))
	);
}
/**
 * 根据具体的应用场景进行调整，因为它们是近似值，并不能保证完全精确。
 * @param x
 * @param y
 * @returns
 */
function transformLng(x: number, y: number): number {
	return (
		300.0 +
		x +
		2.0 * y +
		0.1 * x * x +
		0.1 * x * y +
		0.1 * Math.sqrt(Math.abs(y))
	);
}
