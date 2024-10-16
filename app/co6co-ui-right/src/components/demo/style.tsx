import { defineComponent, CSSProperties } from 'vue';

export default defineComponent({
	name: 'scoped_style_01',
	setup() {
		/**
		 * 样式被编译到了JS文件中
		 */
		const myStyle: CSSProperties = {
			color: 'blue',
			fontSize: '16px',
		};

		return () => <div style={myStyle}>这是一个带有内联样式的文本。</div>;
	},
});
