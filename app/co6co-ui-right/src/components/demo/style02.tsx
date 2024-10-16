import { defineComponent } from 'vue';
import '@/assets/css/demo.css';

export default defineComponent({
	name: 'scoped_style_02',
	setup() {
		/** 此方案使用的是全局样式 */
		return () => <div class="my-class">这是一个带有类名样式的文本。</div>;
	},
});
