import { ref, reactive } from "vue";
import { randomString } from "../../utils";

const appid = import.meta.env.VITE_WX_appid;
//const redirect_uri = import.meta.env.VITE_WX_redirect_uri;
//let stateCode = randomString(18);

const getRedirectUrl = (redirect_uri: string, stateCode: string) => {

    let redirectUrl =
        "https://open.weixin.qq.com/connect/oauth2/authorize?appid=" + appid;
    redirectUrl += "&redirect_uri=" + encodeURI(redirect_uri);
    redirectUrl +=
        "&response_type=code&scope=snsapi_userinfo&state=" +
        stateCode +
        "#wechat_redirect";
    return redirectUrl;
};

export { getRedirectUrl };
