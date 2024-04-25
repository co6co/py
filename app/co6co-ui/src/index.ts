import { intersect } from './module';
export const Greeter = (name: string) => `Hello ${name}`;

//差集
export const Intersect = (a: [], b: []) => intersect(a, b);
