import { ref, reactive } from 'vue';


import { showNotify } from 'vant';
import { randomString } from '../../utils';


const appid = import.meta.env.VITE_WX_appid;
const debug = Boolean(Number(import.meta.env.VITE_IS_DEBUG));

//redirect_uri 应和微信服务器交换获取微信用户信息
//后做304跳转到指定页面
const getScope = (scope: number) => {
	return scope == 0 ? 'snsapi_base' : 'snsapi_userinfo';
};

/** 微信服务器地址 */
const RedirectWxService = (
	redirect_uri: string,
	scope: number,
	stateCode: string
) => {
	console.info('is Debug', debug);
	if (debug)
		return (
			redirect_uri +
			`?code=123456&scope=${getScope(scope)}&state=` +
			encodeURI(stateCode) +
			'#wechat_redirect'
		);
	else {
		let redirectUrl =
			'https://open.weixin.qq.com/connect/oauth2/authorize?appid=' + appid;
		redirectUrl += '&redirect_uri=' + encodeURI(redirect_uri);
		redirectUrl +=
			`&response_type=code&scope=${getScope(scope)}&state=` +
			stateCode +
			'#wechat_redirect';
		return redirectUrl;
	}
};


const getUrl = () => {
	let url = document.location.toString();
	let arrUrl = url.split('//');
	let start = arrUrl[1].indexOf('/');
	let relUrl = arrUrl[1].substring(start); //stop省略，截取从start开始到结尾的所有字符
	if (relUrl.indexOf('?') != -1) {
		relUrl = relUrl.split('?')[0];
	}
    if(debug)relUrl= url 
	return relUrl.replace('#', '**');
};
/** 通过微信服务器地址跳转会 本地服务器地址 */
const redirectUrl = () => {
	showNotify({ type: 'warning', message: `跳转...` });
	const redirect_uri = import.meta.env.VITE_WX_redirect_uri; 
	console.info(redirect_uri)
	const scope = 1;
	let redirectUrl = ''; 
	if (debug) {
		redirectUrl = RedirectWxService(
			redirect_uri,
			scope,
			`${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
		);
	} else {
		let url=`${window.location.protocol}//${window.location.host}${redirect_uri}`
		redirectUrl = RedirectWxService(
			url,
			scope,
			`${randomString(10)}-${scope}-${getUrl()}-${randomString(10)}`
		);
	}
	window.location.href = redirectUrl;
};
export { redirectUrl };
