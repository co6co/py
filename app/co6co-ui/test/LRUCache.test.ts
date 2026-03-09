import LRUCache from '../src/utils/LRUCache';
import assert from 'assert';

// 测试LRUCache类
function testLRUCache() {
    console.log('开始测试LRUCache...');
    
    // 测试1: 基本功能和属性
    console.log('\n测试1: 基本功能和属性');
    const cache = new LRUCache(3);
    assert.strictEqual(cache.capacity, 3, '初始容量应该是3');
    assert.strictEqual(cache.size, 0, '初始大小应该是0');
    console.log('✓ 基本属性测试通过');
    
    // 测试2: 添加元素
    console.log('\n测试2: 添加元素');
    cache.put('a', 1);
    cache.put('b', 2);
    cache.put('c', 3);
    assert.strictEqual(cache.size, 3, '添加3个元素后大小应该是3');
    console.log('✓ 添加元素测试通过');
    
    // 测试3: 访问元素
    console.log('\n测试3: 访问元素');
    assert.strictEqual(cache.get('a'), 1, '获取a应该返回1');
    assert.strictEqual(cache.get('b'), 2, '获取b应该返回2');
    assert.strictEqual(cache.get('c'), 3, '获取c应该返回3');
    console.log('✓ 访问元素测试通过');
    
    // 测试4: 超出容量
    console.log('\n测试4: 超出容量');
    cache.put('d', 4);
    assert.strictEqual(cache.size, 3, '添加第4个元素后大小应该是3');
    assert.strictEqual(cache.has('a'), false, 'a应该被淘汰');
    assert.strictEqual(cache.has('d'), true, 'd应该存在');
    console.log('✓ 超出容量测试通过');
    
    // 测试5: 更新元素
    console.log('\n测试5: 更新元素');
    cache.put('b', 22);
    assert.strictEqual(cache.get('b'), 22, '更新后获取b应该返回22');
    assert.strictEqual(cache.size, 3, '更新后大小应该是3');
    console.log('✓ 更新元素测试通过');
    
    // 测试6: 边界情况
    console.log('\n测试6: 边界情况');
    const emptyCache = new LRUCache(0);
    assert.strictEqual(emptyCache.size, 0, '容量为0的缓存大小应该是0');
    emptyCache.put('test', 'value');
    assert.strictEqual(emptyCache.size, 0, '向容量为0的缓存添加元素后大小应该是0');
    console.log('✓ 边界情况测试通过');
    
    console.log('\n🎉 所有测试通过！');
}

// 运行测试
testLRUCache();
