import Cookie from "js-cookie";
import { Storage } from '../store/Storage'
const authonKey = "Authorization"; 

const storage = new Storage()
export function setToken(token: any,secounds:number=7200) {
    storage.set(authonKey, token,   secounds)
    //localStorage.setItem(authonKey, token);
    return Cookie.set(authonKey, token);
}
export function getToken() {
    
    let token =storage.get(authonKey );
    if (!token) token = getCookie(authonKey);
    //if (!token)  token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDI3MTQ0MDksImlhdCI6MTcwMjYyODAwOSwiaXNzIjoiSldUK1NFUlZJQ0UiLCJkYXRhIjp7ImlkIjoyLCJ1c2VyTmFtZSI6Im9Qd3ZMNkoyWDlZbnl0dW81YWdNTGdLR1ZKUUkiLCJncm91cF9pZCI6Mn19.fztD24fVV5cYtXj1Ev5uoU5bHOGZx01vTkIm-yXdpEM"
    return token;
}
export function removeToken() {
    storage.remove(authonKey) 
    return Cookie.remove(authonKey);
}
export function getCookie(key: string) {
    let data = Cookie.get(key);
    if (data) return data;
    else return "";
}

export function setCookie(key: string, value: string) {
    const val = Cookie.get(key);
    return Cookie.set(key, val || value);
}
export function removeCookie(key: string) {
    return Cookie.remove(key);
}
 