import Cookie from "js-cookie";
const TokenKey = "vue_token";

  
export function setTokes(token:any){
    localStorage.setItem("token",token )
    return Cookie.set("Authorization", token  ); 
}
export function getTokes(){
    let token =localStorage.getItem("token" )
    if (!token) token=getCookie("Authorization")
    if (!token)  token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDI3MTQ0MDksImlhdCI6MTcwMjYyODAwOSwiaXNzIjoiSldUK1NFUlZJQ0UiLCJkYXRhIjp7ImlkIjoyLCJ1c2VyTmFtZSI6Im9Qd3ZMNkoyWDlZbnl0dW81YWdNTGdLR1ZKUUkiLCJncm91cF9pZCI6Mn19.fztD24fVV5cYtXj1Ev5uoU5bHOGZx01vTkIm-yXdpEM"
    return token
}
export function getCookie(key: string) {
    let data=Cookie.get(key);
    if (data)return data
    else return ""
}

export function getToken() {
    return Cookie.get(TokenKey);
}

export function setToken(newToken: any) {
    const token = Cookie.get("token");
    return Cookie.set(TokenKey, token || newToken);
}

export function removeToken() {
    return Cookie.remove(TokenKey);
}
