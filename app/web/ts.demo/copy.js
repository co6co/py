document.addEventListener('copy',(e)=>{
    e.preventDefault()//阻止 默认复制行为
    e.clipboardData.setData('text/plain','不能复制');
    console.info('复制')
})