// copy buttons on code blocks; on phones the step list starts closed
for (const pre of document.querySelectorAll("pre")) {
  const b = document.createElement("button");
  b.textContent = document.body.dataset.copy;
  b.onclick = async () => {
    await navigator.clipboard.writeText(pre.querySelector("code").innerText);
    b.textContent = document.body.dataset.copied;
    setTimeout(() => (b.textContent = document.body.dataset.copy), 1200);
  };
  pre.append(b);
}
if (matchMedia("(max-width: 820px)").matches) document.querySelector("nav details").open = false;
