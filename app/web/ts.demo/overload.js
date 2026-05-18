function createOverload() {
    const fnMap = new Map()
    function overload(...args) {
        const key = args.map(it => typeof it).join(',')
        const fn = fnMap.get(key);
        if (!fn) throw new TypeError('没有找到对应的实现');
        return fn.apply(this, args)
    }
    overload.addImpl = function (...args) {
        const fn = args.pop()
        if (typeof fn !== 'function') throw new TypeError("最后一个参数必须是函数");
        const key = args.join(',')
        fnMap.set(key, fn)
    }
    return overload
}

//  demo
const getUser=createOverload();
getUser.addImpl(()=>{console.log("查询所以用户")})
getUser.addImpl('number',(page,size=10)=>{console.log("按页码和数量查询用户")})
getUser.addImpl('number','number',(page,size=10)=>{console.log("按页码和数量查询用户")})
getUser.addImpl('string',(name='a')=>{console.log("按姓名查询用户")})
getUser.addImpl('string','string',(name,age)=>{console.log("按姓名查询用户")})



