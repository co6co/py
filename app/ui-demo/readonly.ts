type Mutable<T> = {
    -readonly [K in keyof T]: T[K]; // 移除 readonly
};

type RequiredFields<T> = {
    [K in keyof T]-?: T[K]; // 移除 ? (变为必选)
};

type ReadonlyPartial<T> = {
    +readonly [K in keyof T]+?: T[K]; // 添加 readonly 和 ? (默认就是加，+可省略)
};

const user: ReadonlyPartial<{
    name: string;
    age: number;
}> = {
    name: '张三',
    age: 18, 
}; 
//user.age = 20; // ❌ 报错，因为 age 是 readonly 类型
