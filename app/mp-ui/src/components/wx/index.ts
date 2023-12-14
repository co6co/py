import { ref, reactive } from "vue";
import { randomString } from "../../utils";
import alarmTableVue from "../../views/alarmTable.vue";

const appid = import.meta.env.VITE_WX_appid;
//const redirect_uri = import.meta.env.VITE_WX_redirect_uri;
//let stateCode = randomString(18);

//redirect_uri 应和微信服务器交换获取微信用户信息
//后做304跳转到指定页面
const getScope=(scope:number)=>{ 
 return scope==0?"snsapi_base":"snsapi_userinfo"
}
const getRedirectUrl = (redirect_uri: string,scope:number, stateCode: string) => {  
    alert(stateCode)
    return redirect_uri +`?code=123456&scope=${getScope(scope)}&state=` +encodeURI( stateCode )+ "#wechat_redirect"
    let redirectUrl =
        "https://open.weixin.qq.com/connect/oauth2/authorize?appid=" + appid;
    redirectUrl += "&redirect_uri=" + encodeURI(redirect_uri);
    redirectUrl +=
        `&response_type=code&scope=${getScope(scope)}&state=` +
        stateCode +
        "#wechat_redirect";
    return redirectUrl;
};

export { getRedirectUrl };
