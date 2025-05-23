import JSONbig from 'json-bigint';
import dayjs from 'dayjs';
import type { ITree } from '../constants';

export const str2Obj = (str: string) => {
	return JSONbig.parse(str);
};
//需要根据正式项目配置的 env 才能生效 在此使用永久返回 false
//export const isDebug = Boolean(Number(import.meta.env.VITE_IS_DEBUG))

export const sleep = (time: number) => {
	return new Promise((resolve) => setTimeout(resolve, time));
};
/// 创建开始结束时间
export const createStateEndDatetime = (type: number, beforeHour: number) => {
	let endDate: Date | number = new Date();
	let startDate = new Date();
	let times = -1;
	switch (type) {
		case 0:
			endDate = new Date();
			times = endDate.getTime() - beforeHour * 3600 * 1000;
			startDate = new Date(times);
			break;
		case 1:
			startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'));
			endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000;
			break;
		default:
			startDate = new Date(dayjs(new Date()).format('YYYY/MM/DD'));
			endDate = startDate.getTime() + 24 * 3600 * 1000 - 1000;
			break;
	}
	return [
		dayjs(startDate).format('YYYY-MM-DD HH:mm:ss'),
		dayjs(endDate).format('YYYY-MM-DD HH:mm:ss'),
	];
};

/**
 * base64 转 file
 * @param base64
 * @param filename
 * @param contentType
 * @returns File
 */
export const base64ToFile = (
	base64: string,
	filename: string,
	contentType: string
): File => {
	const sliceSize = 512;
	const byteCharacters = atob(base64.split(',')[1]);
	const byteArrays: Array<any> = [];

	for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
		const slice = byteCharacters.slice(offset, offset + sliceSize);

		const byteNumbers = new Array(slice.length);
		for (let i = 0; i < slice.length; i++) {
			byteNumbers[i] = slice.charCodeAt(i);
		}

		const byteArray = new Uint8Array(byteNumbers);
		byteArrays.push(byteArray);
	}

	const blob = new Blob(byteArrays, { type: contentType });
	return new File([blob], filename, { type: contentType });
};
//生成随机字符串
export const randomString = (
	len: number,
	chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
) => {
	let result = '';
	for (let i = len; i > 0; --i)
		result += chars[Math.floor(Math.random() * chars.length)];
	return result;
};
//获取URL 参数
export const getQueryVariable = (key: string) => {
	try {
		//var query = window.location.search.substring(1);
		const query = window.location.href.slice(
			Math.max(0, window.location.href.indexOf('?') + 1)
		);
		const vars = query.split('&');
		for (const var_ of vars) {
			const pair = var_.split('=');
			if (pair[0] == key) {
				return pair[1];
			}
		}
		return null;
	} catch (e) {
		console.error('queryVariable Error:', e);
	}
	return null;
};

export const toggleFullScreen = (
	elem: HTMLElement | any,
	fullScreen = true
) => {
	if (!elem) elem = document.documentElement;
	if (fullScreen) {
		if (elem.requestFullscreen) {
			console.info('1');
			elem.requestFullscreen();
		} else if (elem.mozRequestFullScreen) {
			console.info('2');
			elem.mozRequestFullScreen();
		} else if (elem.webkitRequestFullscreen) {
			console.info('3');
			elem.webkitRequestFullscreen();
		} else if (elem.msRequestFullscreen) {
			console.info('4');
			elem.msRequestFullscreen();
		}
	} else {
		elem = document;
		if (elem.exitFullscreen) {
			elem.exitFullscreen();
		} else if (elem.mozCancelFullScreen) {
			// Firefox
			elem.mozCancelFullScreen();
		} else if (elem.webkitExitFullscreen) {
			// Chrome, Safari and Opera
			elem.webkitExitFullscreen();
		} else if (elem.msExitFullscreen) {
			// Internet Explorer and Edge
			elem.msExitFullscreen();
		}
	}
};

//10进制转16进制补0
export const number2hex = (dec: number, len: number) => {
	let hex = '';
	while (dec) {
		const last = dec & 15;
		hex = String.fromCharCode((last > 9 ? 55 : 48) + last) + hex;
		dec >>= 4;
	}
	if (len) while (hex.length < len) hex = `0${hex}`;
	return hex;
};

type Collection<T> = (a: Array<T>, b: Array<T>) => Array<T>;
// 交集
export const intersect = (array1: [], array2: []) =>
	array1.filter((x) => array2.includes(x));

// 差集
export const minus: Collection<number | string> = (
	array1: Array<number | string>,
	array2: Array<number | string>
) => array1.filter((x) => !array2.includes(x));
// 补集
export const complement = (array1: [], array2: []) => {
	array1
		.filter((v) => {
			return !array2.includes(v);
		})
		.concat(
			array2.filter((v) => {
				return !array1.includes(v);
			})
		);
};
// 并集
export const unionSet = (array1: [], array2: []) => {
	return array1.concat(
		array2.filter((v) => {
			return !array1.includes(v);
		})
	);
};
//key down demo
export const onKeyDown = (e: KeyboardEvent) => {
	console.info('key', e.key);
	if (e.ctrlKey) {
		if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
			//let current = table_module.query.pageIndex.valueOf();
			//let v = e.key == 'ArrowRight' || e.key == 'd' ? current + 1 : current - 1;
			//onPageChange(v);
		}
		if (['ArrowUp', 'ArrowDown'].includes(e.key)) {
			//let current = currentTableItemIndex.value;
			//if (!current) current = 0;
			//let v = e.key == 'ArrowDown' || e.key == 's' ? current + 1 : current - 1;
			//if (0 <= v && v < tableInstance._value.data.length) {
			//	setTableSelectItem(v);
			//} else {
			//	if (v < 0) ElMessage.error('已经是第一条了');
			//	else if (v >= tableInstance._value.data.length) ElMessage.error('已经是最后一条了');
			//}
		}
	}
	//process_view.value.keyDown(e)
	e.stopPropagation();
};

/**
 * 遍历 Tree
 * @param tree
 * @param func  return true 退出循环
 */

export const traverseTreeData = (
	tree: Array<ITree>,
	func: (data: ITree) => void | boolean
) => {
	tree.forEach((data) => {
		data.children && traverseTreeData(data.children, func); // 遍历子树
		const result = func(data);
		if (result) return;
	});
};

export function _convert(
	value: number,
	fromUnit: string,
	toUnit: string,
	conversions: { [key: string]: number }
): number {
	if (!(fromUnit in conversions) || !(toUnit in conversions)) {
		throw new Error('Invalid unit');
	}
	const fromFactor = conversions[fromUnit];
	const toFactor = conversions[toUnit];
	return value * (fromFactor / toFactor);
}

const byteData: { [key: string]: number } = {
	b: 1,
	kb: 1024,
	mb: 1024 ** 2,
	gb: 1024 ** 3,
};
/**
 *
 * @param value
 * @param fromUnit [b|kb|mb|gb]
 * @param toUnit [b|kb|mb|gb]
 * @returns 小数
 */
function convertByte(value: number, fromUnit: string, toUnit: string): number {
	const num = _convert(value, fromUnit, toUnit, byteData);
	return num;
}
/**
 *
 * @param value
 * @param fromUnit [b|kb|mb|gb]
 * @param fixed	保留小数位数 四舍五入
 * @returns
 */
function byte2Unit(value: number, fromUnit: string, fixed: number) {
	let num = 0;
	let unit = 'b';
	let result = 0;
	Object.keys(byteData).forEach((key) => {
		num = convertByte(value, fromUnit, key);
		if (num > 1) {
			unit = key;
			result = num;
		} else return false;
	});
	num = result;
	//// "123.46" (四舍五入)
	return num.toFixed(fixed) + unit.toUpperCase();
}
export { convertByte, byte2Unit };
/**
 *
 * @param value 值
 * @param fromUnit 单位 [m|km|cm|mm|in|ft|yd|mi|nmi]
 * @param toUnit 单位 [m|km|cm|mm|in|ft|yd|mi|nmi]
 * @returns
 */
export function convertDistance(
	value: number,
	fromUnit: string,
	toUnit: string
): number {
	const conversions: { [key: string]: number } = {
		m: 1,
		km: 1000,
		cm: 0.01,
		mm: 0.001,
		in: 0.0254,
		ft: 0.3048,
		yd: 0.9144,
		mi: 1609.34,
		nmi: 1852,
	};
	return _convert(value, fromUnit, toUnit, conversions);
}
