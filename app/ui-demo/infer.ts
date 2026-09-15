// 提取 Promise 里的返回值类型
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;
type Result = UnwrapPromise<Promise<string>>; // string
type Result2 = UnwrapPromise<number>; // number

interface Animal {
    name: string
}
interface Dog extends Animal {
    age: number
}
//结构兼容 Promise(有 then 方法),即使没"继承"Promise
interface Thenable {
    then2<U>(onfulfilled: (v: any) => U): U// : Promise<U>;
}
type A = UnwrapPromise<Thenable>;
interface MyPromise<T> {
    then<R>(onfulfilled: (value: T) => R): MyPromise<R>;
}
type B = UnwrapPromise<MyPromise<number>>; // 匹配 Promise 结构

 