import { defineComponent } from 'vue';
import styles from '@/assets/css/imageVideo.module.less';

export default defineComponent({
	name: 'MyComponent',
	setup() {
		return () => (
			<div class={styles.Image}>这是一个使用 CSS Modules 的文本。</div>
		);
	},
});
