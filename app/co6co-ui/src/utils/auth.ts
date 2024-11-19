import Cookie from 'js-cookie';
import { Storage, SessionKey } from './storage';
import { type IAuthonInfo } from '@/constants';

const authonKey = 'Authorization';
const storage = new Storage();

const refreshToken_key = `${SessionKey}_1`;
const userName_key = `${SessionKey}_name`;

export function setToken(token: any, secounds = 7200) {
	storage.set(authonKey, token, secounds);
}
export function getToken() {
	let token = storage.get(authonKey);
	return token;
}
export function removeToken() {
	storage.remove(authonKey);
	return Cookie.remove(authonKey);
}
export function getCookie(key: string) {
	const data = Cookie.get(key);
	if (data) return data;
	else return '';
}

export function setCookie(key: string, value: string) {
	const val = Cookie.get(key);
	return Cookie.set(key, val || value);
}
export function removeCookie(key: string) {
	return Cookie.remove(key);
}

/**
 * 存储认证信息
 */
export const storeAuthonInfo = (data: IAuthonInfo, userName?: string) => {
	//设置token
	setToken(data.token, data.expireSeconds);
	if (userName) storage.set(userName_key, userName, data.expireSeconds);
	storage.set(SessionKey, data.sessionId, data.expireSeconds);
	if (data.refreshToken)
		storage.set(
			refreshToken_key,
			data.refreshToken.token,
			data.refreshToken.expireSeconds
		);
};
export const removeAuthonInfo = () => {
	removeToken();
	storage.remove(userName_key);
	storage.remove(SessionKey);
	storage.remove(refreshToken_key);
};
/**
 * 获取用户名
 * @returns 返回用户名
 */
export const getUserName = () => {
	return storage.get(userName_key);
};
/**
 * 获取刷新 refreshToken
 * 当 token过期是可使用 refreshToken 刷新 token
 *
 * @returns
 */
export const getRefreshToken = () => {
	return storage.get(refreshToken_key);
};
