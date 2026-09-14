String.prototype.asyncReplaceAll = async function (regex, callback) {
    if (typeof regex == 'string') {
        regex = new RegExp(regex, 'g');
    }
    if (regex instanceof RegExp) {
        if (!regex.global) {
            throw new Error('global must be true');
        }
        regex = new RegExp(regex.source, 'g');
    } else {
        throw new Error('regex must be string or RegExp');
    }
    if (typeof callback != 'function') {
        return this.replaceAll(regex, callback);
    }
    const matches = this.match(regex);
    let targets = matches.map(callback);
    targets = await Promise.all(targets);
    return this.replace(regex, () => {
        return targets.shift();
    });
};
/**
 * 千分位分隔
 * @returns 1,000,000
 */
String.prototype.toThousandsSeparator = function () {
    //1. \B  非单词边界
    //2. (?=...) 不匹配任何实际的字符，只匹配“位置”（零宽匹配）
    //3. (?!\d) 负向零宽断言（Negative Lookahead）

    //const reg = /\B(?=(\d{3})+(?!\d|\.\d))/g;
    const reg = /\B(?=(\d{3})+(?!\d))/g;
    return this.replace(reg, ',');
    //return this.replace(/\B(?=(\d{3})+$)/g, ','); //不支持小数
};
(async () => {
    const str = '1--2-3-12-33';
    r = /\d+/g;
    const result = await str.asyncReplaceAll(/(\d+)/g, async (match) => {
        return 'name' + match;
    });
    console.log(result); // name1--name2-name3-name12-name33
    const result2 = await str.asyncReplaceAll("1", 'name');
    console.log(result2); // name1--name2-name3-name12-name33 
})()

console.log('1000000.336436456'.toThousandsSeparator()); // 1,000,000
console.log('123456789'.toThousandsSeparator()); // 1,000,000