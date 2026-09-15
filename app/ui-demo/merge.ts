function Counter() {}
namespace Counter {
  export let count = 0;
  export function increment() { count++; }
}
Counter()
Counter.increment();
console.log(Counter.count); // 有类型提示