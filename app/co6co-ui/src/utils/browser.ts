import { isClient, isIOS } from '@vueuse/core';

export const isFirefox = (): boolean =>
	isClient && /firefox/i.test(window.navigator.userAgent);

export { isClient, isIOS };

/**
 * @param src public 下的路径
 */
export const loadScript = (src: string, onload: () => void) => {
	const script = document.createElement('script');
	script.src = src; // 注意这里的路径是从public目录开始的相对路径
	script.async = true;
	script.onload = () => {
		if (onload) onload();
		// 可以在这里调用myScript.js中定义的函数等
	};
	script.onerror = () => {
		console.error('Failed to load myScript.js');
	};
	document.body.appendChild(script);
};
/*
 * @param src public 下的路径
 * 移除脚本标签
 */
export const unLoadScript = (src: string) => {
	const scripts = document.querySelectorAll(`script[src="${src}"]`);
	scripts.forEach((script) => document.body.removeChild(script));
};

/**
 * 使用 userAgent 判断是否为手机浏览器
 * @returns 是否是移动端浏览器
 */
export function isMobileBrowser() {
	const userAgent = navigator.userAgent;
	const mobileKeywords = [
		'Android',
		'iPhone',
		'iPad',
		'Windows Phone',
		'BlackBerry',
		'Opera Mini',
	];
	for (const keyword of mobileKeywords) {
		if (userAgent.includes(keyword)) {
			return true;
		}
	}
	return false;
}
