import assert from 'assert';

// 示例测试文件
function testExample() {
    console.log('\n测试示例文件...');
    assert.strictEqual(1 + 1, 2, '1 + 1 应该等于 2');
    assert.strictEqual('hello'.length, 5, 'hello 的长度应该是 5');
    console.log('✓ 示例测试通过');
}

// 运行测试
testExample();
