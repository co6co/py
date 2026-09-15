type Colors = 'red' | 'blue';
type ColorMap = Record<Colors, string>;

// 使用 as 会丢失具体 key 的信息
const badMap = { red: '#f00', blue: '#00f', green: '#0f0' } as ColorMap; // 不报错 但 key被收窄 
// 使用 satisfies 既检查类型，又保留具体字面量
const goodMap = { red: '#f00', blue: '#00f' /*, green: '#0f0'*/ } satisfies ColorMap;// ❌ 报错，对象字面量只能指定已知属性，并且“green”不在类型“ColorMap”中。
// goodMap.green 仍然可以访问，且类型推断为 string 