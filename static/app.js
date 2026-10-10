document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.rating-slider').forEach((slider) => {
    const output = document.getElementById(`${slider.id}-value`);
    const update = () => {
      if (output) output.textContent = slider.value;
      const percent = ((Number(slider.value) - 1) / 4) * 100;
      slider.style.setProperty('--range-progress', `${percent}%`);
    };
    slider.addEventListener('input', update);
    update();
  });

  const form = document.getElementById('assessment-form');
  if (form) {
    form.addEventListener('submit', (event) => {
      const name = form.querySelector('[name="student_name"]');
      if (!name || !name.value.trim()) {
        event.preventDefault();
        name?.focus();
      }
    });
  }
});
