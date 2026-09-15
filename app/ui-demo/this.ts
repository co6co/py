// 方案 1
function greet(this: { name: string }, msg: string) {
    console.log(msg);
    return `${this.name}: ${msg}`;
}

const obj = { name: 'Tom', greet };
obj.greet('Hello'); // ✅ this 指向 obj
//greet('Hello'); // ❌ 直接调用会报错，因为 this 不是预期类型

//方案 2
interface MyObject {
    name: string;
    logName(): void;
}

const obj2: MyObject & ThisType<MyObject> = {
    name: '李四',
    logName() {
        console.log(this.name); // this 被推断为 { name: string }
    }
}; 
obj2.logName(); // ✅ this 指向 obj2 
