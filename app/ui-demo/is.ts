/**
 * `val is string` 是 TypeScript 的 类型谓词语法 。
 * 它告诉编译器：
 *  - 当`isString(val)` 返回`true` 时，`val` 的类型会被收窄为`string` 。
 * @param val 要检查的值
 * @returns 如果值是字符串则返回 true，否则返回 false
 */
function isString(val: any): val is string {
    return typeof val === 'string';
}
function isNonEmptyString(val: any): val is string {
    return typeof val === 'string' && val.length > 0;
}
/**
 * 编译器是不会怀疑你写错的 
 * 它认为会 说这个数字是字符串
 */
function isErrorString(val: any): val is string {
    return typeof val === 'number';
}
function printLength(val: string | number) {
    if (isString(val)) {
        console.log(val.length); // TS 知道这里 val 是 string
    }
}