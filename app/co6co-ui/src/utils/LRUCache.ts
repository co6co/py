/**
 * Least Recently Used，最近最少使用）
 * 是一种缓存淘汰算法，用来在缓存空间不够时
 */
class LRUCache {
    #cache;
    #capacity: number;
    constructor(capacity = 10) {
        this.#capacity = capacity;
        this.#cache = new Map();
    }
    get(key) {
        if (!this.has(key)) {
            return
        }
        const temp = this.#cache.get(key);
        this.#cache.delete(key);
        this.#cache.set(key, temp);
        return temp;
    }
    put(key, value) {
        if (this.#capacity <= 0) {
            return;
        }
        if (this.#cache.has(key)) {
            this.#cache.delete(key);
        }
        console.info(this.#cache.size, this.#capacity)
        if (this.#cache.size >= this.#capacity) {
            this.#cache.delete(this.#cache.keys().next().value);
        }
        this.#cache.set(key, value); 
    }
    has(key) {
        return this.#cache.has(key);
    }
    keys() {
        return this.#cache.keys()
    }
    get capacity() {
        return this.#capacity;
    }
    get size() {
        return this.#cache.size;
    }
    clear() {
        this.#cache.clear();
    }
}
export default LRUCache;