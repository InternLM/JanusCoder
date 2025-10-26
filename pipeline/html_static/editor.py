import os
import re
import json
import random
from typing import *
from utils import print_msgs, chat_api, chat_vlm, convert_image_to_url, gemini_api


Instruction_Prompt_Both = """You are an expert HTML/CSS developer.
You will receive a screenshot and the code of a web page.
Your task is to generate concrete edit instructions for the web page that bring visually noticeable changes to the page. An edit instruction is composed of an edit action, a visible UI element, and an edit attribute.

The edit action types include:
(1) Add (introducing new UI elements)
(2) Change (modifying elements)
(3) Delete (removing elements)

The editable UI elements include:
(1) Button (clickable element for user actions, e.g., "Submit", "Save")
(2) Input field (form element for text or data entry, such as textboxes or number inputs)
(3) Card (container element for grouping related content, often with a border or shadow)
(4) List item (individual entry within a list, such as menu or todo items)
(5) Divider (horizontal or vertical line used to separate content sections)
(6) Heading (text element indicating section titles, e.g., <h1>, <h2>)
(7) Navigation bar (top-level menus and links)  
(8) Image (pictures, logos, or illustrations)  
(9) Icon (symbolic graphic, e.g., checkmark, star)  
(10) Table (rows and columns of data) 

The editable attribute types include:
(1) text (including content, font, and typography modifications)
(2) color (encompassing background colors, text colors, and accent colors)
(3) position (spatial arrangement and layout adjustments)
(4) size (dimensional scaling and resizing operations)
(5) shape (geometric modifications and structural changes)
(6) layout & spacing (holistic modifications affecting entire UI components)

---

## Requirements for Generating Edit Instructions
1. **Visual Impact** 
Every instruction must produce a clear, visually noticeable change (e.g., layout restructuring, color scheme shifts, adding or removing visible components).

2. **Visual References Only** 
Always describe target elements by their appearance or position on the page (e.g., "the large green button at the bottom right", "the navigation bar at the top"). Never use code-specific terms like class names, IDs, or HTML tags.

3. **High-Level Intentions**
Express edits as general intentions rather than precise technical details (e.g., say "move the button closer to the edge" instead of "move the button by 10px").

4. **No Interactivity**
Exclude interactive behaviors such as hover states, animations, or JavaScript-based actions.

5. **Screenshot-Grounded Only**
Do not mention information that could only be known from inspecting the HTML/CSS source. Rely solely on what is visible in the screenshot.

6. **Element Relationships or Multi-Property Changes**
An instruction must either:
- Involve at least two elements in relation to each other (e.g., alignment, grouping, ordering, spacing), or
- Combine multiple changes to a single element into one instruction (e.g., "make the card smaller and add a gray border").

7. **No Redundancy**
Avoid overly similar or repetitive instructions (e.g., do not output both "Swap the first and second buttons" and "Swap the third and fourth buttons").

8. **Output Format**
Generate 3 to 5 instructions as a numbered list, with no explanations or extra comments. If no suitable instruction can be generated, output exactly one word: "None".

---

## Code
```html
{code}
```

## Output Instructions
"""


Instruction_Prompt_Image = """You are an expert HTML/CSS developer.
You will receive a screenshot of a web page.
Your task is to generate concrete edit instructions for the web page that bring visually noticeable changes to the page. An edit instruction is composed of an edit action, a visible UI element, and an edit attribute.

The edit action types include:
(1) Add (introducing new UI elements)
(2) Change (modifying elements)
(3) Delete (removing elements)

The editable UI elements include:
(1) Button (clickable element for user actions, e.g., "Submit", "Save")
(2) Input field (form element for text or data entry, such as textboxes or number inputs)
(3) Card (container element for grouping related content, often with a border or shadow)
(4) List item (individual entry within a list, such as menu or todo items)
(5) Divider (horizontal or vertical line used to separate content sections)
(6) Heading (text element indicating section titles, e.g., <h1>, <h2>)
(7) Navigation bar (top-level menus and links)  
(8) Image (pictures, logos, or illustrations)  
(9) Icon (symbolic graphic, e.g., checkmark, star)  
(10) Table (rows and columns of data) 

The editable attribute types include:
(1) text (including content, font, and typography modifications)
(2) color (encompassing background colors, text colors, and accent colors)
(3) position (spatial arrangement and layout adjustments)
(4) size (dimensional scaling and resizing operations)
(5) shape (geometric modifications and structural changes)
(6) layout & spacing (holistic modifications affecting entire UI components)

---

## Requirements for Generating Edit Instructions
1. **Visual Impact** 
Every instruction must produce a clear, visually noticeable change (e.g., layout restructuring, color scheme shifts, adding or removing visible components).

2. **Visual References Only** 
Always describe target elements by their appearance or position on the page (e.g., "the large green button at the bottom right", "the navigation bar at the top"). Never use code-specific terms like class names, IDs, or HTML tags.

3. **High-Level Intentions**
Express edits as general intentions rather than precise technical details (e.g., say "move the button closer to the edge" instead of "move the button by 10px").

4. **No Interactivity**
Exclude interactive behaviors such as hover states, animations, or JavaScript-based actions.

5. **Screenshot-Grounded Only**
Do not mention information that could only be known from inspecting the HTML/CSS source. Rely solely on what is visible in the screenshot.

6. **Element Relationships or Multi-Property Changes**
An instruction must either:
- Involve at least two elements in relation to each other (e.g., alignment, grouping, ordering, spacing), or
- Combine multiple changes to a single element into one instruction (e.g., "make the card smaller and add a gray border").

7. **No Redundancy**
Avoid overly similar or repetitive instructions (e.g., do not output both "Swap the first and second buttons" and "Swap the third and fourth buttons").

8. **Output Format**
Generate 3 to 5 instructions as a numbered list, with no explanations or extra comments. If no suitable instruction can be generated, output exactly one word: "None".

---

## Output Instructions
"""

Edit_Prompt = """You are an expert HTML/CSS developer. 
You take a piece of code of a reference web page, and an instruction from the user.
You need to modify the code according to the user's instruction to make the webpage satisfy user's demands.

Requirements:
- Do not modify any part of the web page other than the parts covered by the instructions.
- For images, use placeholder images from https://placehold.co
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.

You MUST wrap your entire code output inside the following markdown fences: ```html and ```.

Do not output any extra information or comments.

---

## Example Instruction
Add a frame to each listing under #TrendingItems to ensure format match and vertical alignment

## Example Code
```html
<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Creeps Store</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            color: #333;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        /* Header */
        header {
            background: #fff;
            border-bottom: 1px solid #eee;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0;
            font-size: 14px;
        }
        .top-bar .search input {
            padding: 8px;
            width: 200px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .top-bar .logo {
            font-size: 24px;
            font-weight: bold;
            font-style: italic;
            color: #000;
            text-align: center  center;
        }
        .top-bar .auth-links a {
            margin-left: 10px;
        }
        nav {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px 0;
        }
        .nav-menu {
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
            width: 100%;
            justify-content: space-around;
        }
        .nav-menu li {
            flex: 1;
            text-align: center;
        }
        .nav-menu li a {
            padding: 10px;
            display: block;
        }
        .nav-menu li a:hover, .nav-menu li a:focus {
            background: #f5f5f5;
        }
        /* Promo Banner */
        .promo {
            background: #f9f9f9;
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #555;
        }
        /* Trending Items */
        #trending-items {
            padding: 40px 0;
            text-align: center;
        }
        #trending-items h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .product-card {
            text-align: center;
        }
        .placeholder-img {
            width: 100%;
            height: 200px;
            background: #ccc;
            margin-bottom: 10px;
        }
        .product-card p {
            margin: 5px 0;
        }
        .product-card button {
            background: #333;
            color: #fff;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            margin-top: 10px;
        }
        .product-card button:hover {
            background: #555;
        }
        .pagination {
            text-align: center;
            margin-top: 20px;
        }
        .pagination a {
            margin: 0 5px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .pagination a:hover {
            background: #f5f5f5;
        }
        /* Customer Share */
        #customer-share {
            padding: 40px 0;
            text-align: center;
            background: #f9f9f9;
        }
        #customer-share h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        /* Illustrator Collab */
        #illustrator-collab {
            padding: 40px 0;
            text-align: center;
        }
        #illustrator-collab h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        .see-more {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            border: 1px solid #333;
            border-radius: 5px;
        }
        .see-more:hover {
            background: #f5f5f5;
        }
        /* Footer */
        footer {
            background: #333;
            color: #fff;
            padding: 40px 0;
        }
        .footer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .footer-grid > div {
            padding: 20px;
            box-sizing: border-box;
        }
        .footer-grid h3 {
            font-size: 18px;
            margin-bottom: 10px;
        }
        .footer-grid ul {
            list-style: none;
            padding: 0;
        }
        .footer-grid ul li {
            margin-bottom: 5px;
        }
        .newsletter .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .newsletter input[type="email"] {
            padding: 10px;
            flex: 1;
        }
        .newsletter button {
            background: #555;
            color: #fff;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
        }
        .newsletter button:hover {
            background: #777;
        }
        .newsletter .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .connect-us p {
            margin: 5px 0;
        }
        .copyright {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .nav-menu {
                flex-direction: column;
                align-items: center;
            }
            .nav-menu li {
                margin: 10px 0;
                flex: none;
            }
            .top-bar {
                flex-direction: column;
                align-items: stretch;
                gap: 10px;
            }
            .top-bar .search input {
                width: 100%;
            }
            .top-bar .logo {
                order: -1;
            }
            .top-bar .auth-links {
                text-align: right;
            }
            .top-bar .auth-links a {
                margin: 5px 0;
                display: inline-block;
            }
            .newsletter .input-group {
                flex-direction: column;
            }
            .newsletter .input-group input,
            .newsletter .input-group button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="top-bar">
                <div class="search">
                    <input type="text" placeholder="Search items">
                </div>
                <div class="logo">CREEPS</div>
                <div class="auth-links">
                    <a href="#login">登入</a>
                    <a href="#points">查看點數</a>
                </div>
            </div>
            <nav>
                <ul class="nav-menu" role="navigation">
                    <li><a href="#home">Home</a></li>
                    <li><a href="#best-sellers">Best Sellers</a></li>
                    <li><a href="#collaboration">Collaboration</a></li>
                    <li><a href="#t-shirts">T-shirts</a></li>
                    <li><a href="#hoodies">Hoodies</a></li>
                    <li><a href="#other-products">Other Products</a></li>
                    <li><a href="#cashback">會員現金回贈計劃</a></li>
                    <li><a href="#more">更多</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <section class="promo">
        <div class="container">
            <p>>>加入會員|下單享有5－10％回贈金額, 可於下張訂單作為現金扣減<<</p>
        </div>
    </section>

    <section id="trending-items">
        <div class="container">
            <h2>#TrendingItems</h2>
            <div class="product-grid" id="product-grid"></div>
            <div class="pagination">
                <a href="#">1</a>
                <a href="#">2</a>
                <a href="#">3</a>
                <a href="#">4</a>
            </div>
        </div>
    </section>

    <section id="customer-share">
        <div class="container">
            <h2>#CustomerShare</h2>
            <div class="placeholder-img"></div>
        </div>
    </section>

    <section id="illustrator-collab">
        <div class="container">
            <h2>#IllustratorCollab</h2>
            <div class="placeholder-img"></div>
            <a href="#" class="see-more">See More</a>
        </div>
    </section>

    <footer>
        <div class="container">
            <div class="footer-grid">
                <div class="sitemap">
                    <h3>Sitemap</h3>
                    <ul>
                        <li><a href="#about-us">About us</a></li>
                        <li><a href="#t-shirts">T-Shirts</a></li>
                        <li><a href="#hoodies">Hoodies</a></li>
                        <li><a href="#phone-case">Phone Case</a></li>
                        <li><a href="#bottoms">Bottoms</a></li>
                        <li><a href="#jacket">Jacket</a></li>
                        <li><a href="#accessories">Accessories</a></li>
                        <li><a href="#faq">FAQ</a></li>
                    </ul>
                </div>
                <div class="newsletter">
                    <h3>Newsletter*</h3>
                    <div class="input-group">
                        <input type="email" placeholder="Your email">
                        <button>Subscribe</button>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="subscribe">
                        <label for="subscribe">I want to subscribe to your mailing list.</label>
                    </div>
                </div>
                <div class="connect-us">
                    <h3>Connect Us</h3>
                    <p>合作銷售點 ( #CCCCCc Se/ect 選物店)</p>
                    <p>旺角/朗豪坊L502號鋪</p>
                    <p>中環/中環街市116號鋪</p>
                    <p>將軍澳/東港城135號鋪</p>
                    <p>尖沙咀/ The One L601號鋪</p>
                    <p>門市地址:</p>
                    <p>旺角花園街75 - 77號花園商業大廈四樓 01 - 02 室</p>
                    <p>營業時間 ｜16：00 － 20：00 (星期一、四休息）</p>
                </div>
            </div>
            <div class="copyright">
                <p>© 2024 CREEPS STORE All Rights Reserved.</p>
            </div>
        </div>
    </footer>

    <script>
        // Product data
        const products = [
            { name: "Really Like You Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Blue Mood Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Prelude Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Silver Girl Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Mah Jong Wins Oversized Half-sleeve T-shirt", price: "HK$148.00", colors: 4 },
            { name: "黑玻璃 Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
        ];

        // Generate product cards
        const productGrid = document.getElementById('product-grid');
        products.forEach(product => {
            const card = document.createElement('div');
            card.className = 'product-card';
            card.innerHTML = `
                <div class="placeholder-img"></div>
                <p>${product.colors} colors</p>
                <p>快速瀏覽</p>
                <p>【Creeps Original】${product.name}</p>
                <p>價格${product.price}</p>
                <button>Add to Cart</button>
            `;
            productGrid.appendChild(card);
        });

        // Tab navigation accessibility
        const navLinks = document.querySelectorAll('.nav-menu a');
        navLinks.forEach(link => {
            link.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const currentIndex = Array.from(navLinks).indexOf(e.target);
                    const nextIndex = (currentIndex + 1) % navLinks.length;
                    navLinks[nextIndex].focus();
                }
            });
        });
    </script>
</body>
</html>
```

## Example Output
```html
<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Creeps Store</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            color: #333;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        /* Header */
        header {
            background: #fff;
            border-bottom: 1px solid #eee;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0;
            font-size: 14px;
        }
        .top-bar .search input {
            padding: 8px;
            width: 200px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .top-bar .logo {
            font-size: 24px;
            font-weight: bold;
            font-style: italic;
            color: #000;
            text-align: center;
        }
        .top-bar .auth-links a {
            margin-left: 10px;
        }
        nav {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px 0;
        }
        .nav-menu {
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
            width: 100%;
            justify-content: space-around;
        }
        .nav-menu li {
            flex: 1;
            text-align: center;
        }
        .nav-menu li a {
            padding: 10px;
            display: block;
        }
        .nav-menu li a:hover, .nav-menu li a:focus {
            background: #f5f5f5;
        }
        /* Promo Banner */
        .promo {
            background: #f9f9f9;
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #555;
        }
        /* Trending Items */
        #trending-items {
            padding: 40px 0;
            text-align: center;
        }
        #trending-items h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .product-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            height: 360px;
            text-align: center;
            box-sizing: border-box;
        }
        .placeholder-img {
            width: 100%;
            height: 200px;
            background: #ccc;
            margin-bottom: 10px;
        }
        .product-card p {
            margin: 5px 0;
            flex-grow: 0;
        }
        .product-card button {
            background: #333;
            color: #fff;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            margin-top: auto;
            border-radius: 4px;
        }
        .product-card button:hover {
            background: #555;
        }
        .pagination {
            text-align: center;
            margin-top: 20px;
        }
        .pagination a {
            margin: 0 5px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .pagination a:hover {
            background: #f5f5f5;
        }
        /* Customer Share */
        #customer-share {
            padding: 40px 0;
            text-align: center;
            background: #f9f9f9;
        }
        #customer-share h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        /* Illustrator Collab */
        #illustrator-collab {
            padding: 40px 0;
            text-align: center;
        }
        #illustrator-collab h2 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        .see-more {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            border: 1px solid #333;
            border-radius: 5px;
        }
        .see-more:hover {
            background: #f5f5f5;
        }
        /* Footer */
        footer {
            background: #333;
            color: #fff;
            padding: 40px 0;
        }
        .footer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .footer-grid > div {
            padding: 20px;
            box-sizing: border-box;
        }
        .footer-grid h3 {
            font-size: 18px;
            margin-bottom: 10px;
        }
        .footer-grid ul {
            list-style: none;
            padding: 0;
        }
        .footer-grid ul li {
            margin-bottom: 5px;
        }
        .newsletter .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .newsletter input[type="email"] {
            padding: 10px;
            flex: 1;
        }
        .newsletter button {
            background: #555;
            color: #fff;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
        }
        .newsletter button:hover {
            background: #777;
        }
        .newsletter .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .connect-us p {
            margin: 5px 0;
        }
        .copyright {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .nav-menu {
                flex-direction: column;
                align-items: center;
            }
            .nav-menu li {
                margin: 10px 0;
                flex: none;
            }
            .top-bar {
                flex-direction: column;
                align-items: stretch;
                gap: 10px;
            }
            .top-bar .search input {
                width: 100%;
            }
            .top-bar .logo {
                order: -1;
            }
            .top-bar .auth-links {
                text-align: right;
            }
            .top-bar .auth-links a {
                margin: 5px 0;
                display: inline-block;
            }
            .newsletter .input-group {
                flex-direction: column;
            }
            .newsletter .input-group input,
            .newsletter .input-group button {
                width: 100%;
            }
            .product-card {
                height: auto;
                min-height: 360px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="top-bar">
                <div class="search">
                    <input type="text" placeholder="Search items">
                </div>
                <div class="logo">CREEPS</div>
                <div class="auth-links">
                    <a href="#login">登入</a>
                    <a href="#points">查看點數</a>
                </div>
            </div>
            <nav>
                <ul class="nav-menu" role="navigation">
                    <li><a href="#home">Home</a></li>
                    <li><a href="#best-sellers">Best Sellers</a></li>
                    <li><a href="#collaboration">Collaboration</a></li>
                    <li><a href="#t-shirts">T-shirts</a></li>
                    <li><a href="#hoodies">Hoodies</a></li>
                    <li><a href="#other-products">Other Products</a></li>
                    <li><a href="#cashback">會員現金回贈計劃</a></li>
                    <li><a href="#more">更多</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <section class="promo">
        <div class="container">
            <p>>>加入會員|下單享有5－10％回贈金額, 可於下張訂單作為現金扣減<<</p>
        </div>
    </section>

    <section id="trending-items">
        <div class="container">
            <h2>#TrendingItems</h2>
            <div class="product-grid" id="product-grid"></div>
            <div class="pagination">
                <a href="#">1</a>
                <a href="#">2</a>
                <a href="#">3</a>
                <a href="#">4</a>
            </div>
        </div>
    </section>

    <section id="customer-share">
        <div class="container">
            <h2>#CustomerShare</h2>
            <div class="placeholder-img"></div>
        </div>
    </section>

    <section id="illustrator-collab">
        <div class="container">
            <h2>#IllustratorCollab</h2>
            <div class="placeholder-img"></div>
            <a href="#" class="see-more">See More</a>
        </div>
    </section>

    <footer>
        <div class="container">
            <div class="footer-grid">
                <div class="sitemap">
                    <h3>Sitemap</h3>
                    <ul>
                        <li><a href="#about-us">About us</a></li>
                        <li><a href="#t-shirts">T-Shirts</a></li>
                        <li><a href="#hoodies">Hoodies</a></li>
                        <li><a href="#phone-case">Phone Case</a></li>
                        <li><a href="#bottoms">Bottoms</a></li>
                        <li><a href="#jacket">Jacket</a></li>
                        <li><a href="#accessories">Accessories</a></li>
                        <li><a href="#faq">FAQ</a></li>
                    </ul>
                </div>
                <div class="newsletter">
                    <h3>Newsletter*</h3>
                    <div class="input-group">
                        <input type="email" placeholder="Your email">
                        <button>Subscribe</button>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="subscribe">
                        <label for="subscribe">I want to subscribe to your mailing list.</label>
                    </div>
                </div>
                <div class="connect-us">
                    <h3>Connect Us</h3>
                    <p>合作銷售點 ( #CCCCCc Se/ect 選物店)</p>
                    <p>旺角/朗豪坊L502號鋪</p>
                    <p>中環/中環街市116號鋪</p>
                    <p>將軍澳/東港城135號鋪</p>
                    <p>尖沙咀/ The One L601號鋪</p>
                    <p>門市地址:</p>
                    <p>旺角花園街75 - 77號花園商業大廈四樓 01 - 02 室</p>
                    <p>營業時間 ｜16：00 － 20：00 (星期一、四休息）</p>
                </div>
            </div>
            <div class="copyright">
                <p>© 2024 CREEPS STORE All Rights Reserved.</p>
            </div>
        </div>
    </footer>

    <script>
        // Product data
        const products = [
            { name: "Really Like You Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Blue Mood Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Prelude Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Silver Girl Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
            { name: "Mah Jong Wins Oversized Half-sleeve T-shirt", price: "HK$148.00", colors: 4 },
            { name: "黑玻璃 Oversized Printed T-shirt", price: "HK$148.00", colors: 6 },
        ];

        // Generate product cards
        const productGrid = document.getElementById('product-grid');
        products.forEach(product => {
            const card = document.createElement('div');
            card.className = 'product-card';
            card.innerHTML = `
                <div class="placeholder-img"></div>
                <p>${product.colors} colors</p>
                <p>快速瀏覽</p>
                <p>【Creeps Original】${product.name}</p>
                <p>價格${product.price}</p>
                <button>Add to Cart</button>
            `;
            productGrid.appendChild(card);
        });

        // Tab navigation accessibility
        const navLinks = document.querySelectorAll('.nav-menu a');
        navLinks.forEach(link => {
            link.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const currentIndex = Array.from(navLinks).indexOf(e.target);
                    const nextIndex = (currentIndex + 1) % navLinks.length;
                    navLinks[nextIndex].focus();
                }
            });
        });
    </script>
</body>
</html>
```

---

"""


Judge_Prompt_Both = """
You are a **Senior Quality Assurance Expert in AI-Generated HTML/CSS Code Editing and Visualization**. 
Your mission is to provide a rigorous, objective, and multi-faceted evaluation of AI-generated code modification tasks. 
You will be given: 
1. the original rendered `Image` (the first input image), 
2. the modified rendered `Image` (the second input image), 
3. the natural language `Instruction` (user's command for modification),
4. the original `Code`,
5. the modified `Code`.


Your evaluation must follow a detailed **Chain of Thought** process, analyzing each component before assigning a score.

-----

### **Evaluation Framework**

**Stage 1: Comprehensive Task Understanding**

  * **Analyze the Instruction:** Break down the user's request into explicit requirements (e.g., "change background to blue", "add a red button", "remove the chart title") and implicit requirements (e.g., style consistency, element positioning).
  * **Compare Images:** Identify what has changed between the original and modified image. List all observed modifications.
  * **Compare Codes:** Identify what has changed between the original and modified code. List all observed modifications.
  * **Match Against Instruction:** Verify whether the observed code and image modifications directly and fully correspond to the instruction. Check if there are missing elements, extra unintended changes, or partial compliance.

**Stage 2: Multi-dimensional Rating & Scoring**
Based on your analysis, you will rate the given example across four dimensions. Then, you will provide a final score based on the detailed guidelines below.

#### **Evaluation Dimensions**

1.  **Instruction Fulfillment**

      * **Accuracy:** Does the modified code and its rendered image correctly implement every requested change?
      * **Completeness:** Are all aspects of the instruction covered without omissions?

2.  **Modification Precision**

      * **Unintended Changes:** Were there any modifications not requested by the instruction?
      * **Faithfulness:** Did the modification preserve all unrelated elements from the original code and image?

3.  **Visual Quality & Consistency**

      * **Clarity:** Is the modified element clear, readable, and well-rendered?
      * **Consistency:** Does the change blend naturally with the rest of the image (no layout break, no visual artifacts)?

4.  **Code Quality & Consistency**

      * **Correctness:** Is the modified code syntactically correct and able to render without errors?
      * **Structure & Readability:** Does the code remain organized, maintainable, and easy to understand? Are proper naming conventions, indentation, and commenting preserved?
      * **Efficiency:** Are modifications implemented in a logical, efficient, and minimally invasive way, without unnecessary complexity or code duplication?
      * **Consistency:** Are the coding style and structure consistent with the original codebase (unless the instruction specifies otherwise)?

5.  **Task Relevance & Usefulness**

      * **Practicality:** Does the instruction represent a realistic and useful web-editing scenario?
      * **Value:** Is this example a good benchmark for evaluating AI code-editing and web UI understanding capabilities?

#### **Scoring Guidelines (1-5 Scale)**

  * **5 (Excellent):** All instructions perfectly implemented; no extra changes; code and visuals are clean and consistent, code quality is high.
  * **4 (Good):** Instruction mostly implemented with only minor imperfections or negligible extra changes. Code and visuals are generally high quality.
  * **3 (Fair):** Some parts of the instruction are missing or incorrectly applied; noticeable issues in code, visuals, or unintended changes.
  * **2 (Poor):** Major deviation from the instruction; significant missing or wrong modifications; poor code or visual quality.
  * **1 (Failed):** Instruction not followed at all, or modifications are irrelevant/incorrect; code may be broken or non-renderable

-----

### **Output Specifications**

Your final output must be a single JSON object. It must include your detailed `Chain of Thought` reasoning, a score for each of the four dimensions, and a final `Total Score` (the average of the dimensional scores). If you give a score of 5, you must explicitly state that all requirements are perfectly satisfied. If you give a score below 5, you must list which requirements are violated.

-----

### **Illustrative Example**

**Input Data:**
  * **Original Image:** [A screenshot of the original HTML page. In the real input, it is the first image.]
  * **Modified Image:** [A screenshot of the modified HTML page. In the real input, it is the second image.]
  * **Instruction:** Position the address box alongside the sidebar menu and adjust the text color inside the box to match the main text color for consistency.
  * **Original Code:** [The original HTML code.]
  * **Modified Code:** [The modified HTML code.]
  

**Output:**

```json
{{
  "Chain of Thought": "The instruction requires positioning the address box alongside the sidebar menu and ensuring its text color matches the main text. In the original page, the address box is centered above the content, not aligned with the sidebar. In the modified version, the address box appears at the top right, visually next to the sidebar menu. The text color remains black, matching the main content. The implementation uses absolute positioning, achieving a two-column layout but with some alignment and responsiveness limitations. No unrelated elements are changed. The result visually fulfills the instruction with minor technical and aesthetic compromises.",
  "Instruction Fulfillment": 4,
  "Modification Precision": 5,
  "Visual Quality & Consistency": 4,
  "Code Quality & Consistency": 4,
  "Task Relevance & Usefulness": 5,
  "Total Score": 4
}}
```

-----

### Instruction
{instruction}

### Original Code
```html
{vanilla_code}
```

### Modified Code
```html
{modified_code}
```

### Output
"""

Judge_Prompt_Image = """
You are a **Senior Quality Assurance Expert in AI-Generated HTML/CSS Code Editing and Visualization**. 
Your mission is to provide a rigorous, objective, and multi-faceted evaluation of AI-generated code modification tasks. 
You will be given: 
1. the original rendered `Image` (the first input image), 
2. the modified rendered `Image` (the second input image), 
3. the natural language `Instruction` (user's command for modification),


Your evaluation must follow a detailed **Chain of Thought** process, analyzing each component before assigning a score.

-----

### **Evaluation Framework**

**Stage 1: Comprehensive Task Understanding**

  * **Analyze the Instruction:** Break down the user's request into explicit requirements (e.g., "change background to blue", "add a red button", "remove the chart title") and implicit requirements (e.g., style consistency, element positioning).
  * **Compare Images:** Identify what has changed between the original and modified image. List all observed modifications.
  * **Match Against Instruction:** Verify whether the observed image modifications directly and fully correspond to the instruction. Check if there are missing elements, extra unintended changes, or partial compliance.

**Stage 2: Multi-dimensional Rating & Scoring**
Based on your analysis, you will rate the given example across five dimensions. Then, you will provide a final score based on the detailed guidelines below.

#### **Evaluation Dimensions**

1.  **Instruction Fulfillment**

      * **Accuracy:** Does the modified code and its rendered image correctly implement every requested change?
      * **Completeness:** Are all aspects of the instruction covered without omissions?

2.  **Modification Precision**

      * **Unintended Changes:** Were there any modifications not requested by the instruction?
      * **Minimal Necessary Change:** Was the change scope minimized to only what was required, avoiding collateral edits?

3.  **Modification Recall**

      * **Faithfulness:** Did the modification preserve all unrelated elements from the original code and image?
      * **No Content Loss:** Was any original information, layout, or visual element inadvertently lost, degraded, or corrupted?

4.  **Visual Quality & Consistency**

      * **Clarity:** Is the modified element clear, readable, and well-rendered?
      * **Consistency:** Does the change blend naturally with the rest of the image (no layout break, no visual artifacts)?

5.  **Task Relevance & Usefulness**

      * **Practicality:** Does the instruction represent a realistic and useful web-editing scenario?
      * **Value:** Is this example a good benchmark for evaluating AI code-editing and web UI understanding capabilities?

#### **Scoring Guidelines (1-5 Scale)**

  * **5 (Excellent):** All instructions perfectly implemented; no extra changes; code and visuals are clean and consistent, code quality is high.
  * **4 (Good):** Instruction mostly implemented with only minor imperfections or negligible extra changes. Code and visuals are generally high quality.
  * **3 (Fair):** Some parts of the instruction are missing or incorrectly applied; noticeable issues in code, visuals, or unintended changes.
  * **2 (Poor):** Major deviation from the instruction; significant missing or wrong modifications; poor code or visual quality.
  * **1 (Failed):** Instruction not followed at all, or modifications are irrelevant/incorrect; code may be broken or non-renderable

-----

### **Output Specifications**

  * Your final output must be a single JSON object. It must include your detailed `Chain of Thought` reasoning, a score for each of the five dimensions, and a final `Total Score`. 
  * The `Total Score` should reflect your holistic, overall judgment of the result as a whole, not a simple arithmetic average of the five dimension scores. 
  * If you give a score of 5, you must explicitly state that all requirements are perfectly satisfied. If you give a score below 5, you must list which requirements are violated. 
  * All scores for each criterion must be integers (1, 2, 3, 4, or 5). Do not assign fractional or decimal scores to any item, including the overall score.

-----

### **Illustrative Example**

**Input Data:**
  * **Original Image:** [A screenshot of the original HTML page. In the real input, it is the first image.]
  * **Modified Image:** [A screenshot of the modified HTML page. In the real input, it is the second image.]
  * **Instruction:** Position the address box alongside the sidebar menu and adjust the text color inside the box to match the main text color for consistency.
  

**Output:**

```json
{{
  "Chain of Thought": "The instruction requires positioning the address box alongside the sidebar menu and ensuring its text color matches the main text. In the original page, the address box is centered above the content, not aligned with the sidebar. In the modified version, the address box appears at the top right, visually next to the sidebar menu. The text color remains black, matching the main content. The implementation uses absolute positioning, achieving a two-column layout but with some alignment and responsiveness limitations. No unrelated elements are changed. The result visually fulfills the instruction with minor technical and aesthetic compromises.",
  "Instruction Fulfillment": 4,
  "Modification Precision": 5,
  "Modification Recall": 5,
  "Visual Quality & Consistency": 4,
  "Task Relevance & Usefulness": 5,
  "Total Score": 4
}}
```

-----

### Instruction
{instruction}

```

### Output
"""



def extract_instructions(text: str):
    pattern = r'(?m)^(\d+)\.\s'

    splits = [m.start() for m in re.finditer(pattern, text)]
    splits.append(len(text))  

    instructions = [text[splits[i]:splits[i+1]].strip() for i in range(len(splits) - 1)]

    instructions = [re.sub(r'^\d+\.\s', '', instr).strip() for instr in instructions]

    return instructions

def extract_html_code(text: str):
    return text.replace("```html", "").replace("```", "").strip()


def generate(code: str, image_path: str, mode: str) -> Tuple[Dict, str]:
    chat_history = []

    image_url = convert_image_to_url(image_path)
    
    if mode == "both":
        prompt = Instruction_Prompt_Both.format(code=code)
    elif mode == "image":
        prompt = Instruction_Prompt_Image.format()
    else:
        assert 0, f"invalid mode: {mode}"
    
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt},
    ]
    msgs = [{"role":"user", "content": content}]
    response = chat_vlm(msgs, max_tokens=1000)
    instruction_list = extract_instructions(response)
    instruction = random.choice(instruction_list)
    chat_history.append({"role":"user", "content": prompt})
    chat_history.append({"role":"assistant", "content": response})


    prompt = Edit_Prompt + f"## Instruction:\n{instruction}\n\n## Code:\n```html\n{code}\n```\n\n## Output:\n"
    msgs = [{"role": "user", "content": prompt}]
    response = chat_api(msgs, max_tokens=30000, temperature=1.0)
    modified_code = extract_html_code(response)
    chat_history.append({"role":"user", "content": prompt})
    chat_history.append({"role":"assistant", "content": response})
    
    return {"human": instruction}, modified_code, chat_history

def verify(
    vanilla_code: str, 
    vanilla_image: str, 
    modified_code: str, 
    modified_image: str, 
    instruction: str,
    chat_history: List,
    mode: str,
) -> Dict:
    image_1 = convert_image_to_url(vanilla_image)
    image_2 = convert_image_to_url(modified_image)
    if mode == "both":
        prompt = Judge_Prompt_Both.format(instruction=instruction, vanilla_code=vanilla_code, modified_code=modified_code)
    elif mode == "image":
        prompt = Judge_Prompt_Image.format(instruction=instruction)
    else:
        assert 0, f"invalid mode: {mode}"
    content = [
        {"type": "image_url", "image_url": {"url": image_1}},
        {"type": "image_url", "image_url": {"url": image_2}},
        {"type": "text", "text": prompt},
    ]
    msgs = [{"role":"user", "content": content}]
    response = chat_vlm(msgs, max_tokens=2000)
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": response})
    judgement = json.loads(response.replace("```json", "").replace("```", "").strip())

    return judgement

