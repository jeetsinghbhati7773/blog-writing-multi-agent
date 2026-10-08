from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.ui.components import (
    render_article_metrics,
    render_content_brief,
    render_workflow_progress,
)
from src.ui.helpers import (
    bundle_zip,
    extract_title_from_md,
    images_zip,
    list_past_blogs,
    read_md_file,
    safe_slug,
    export_to_markdown_with_frontmatter,
    export_to_html,
    export_to_json,
)
from src.agent.tools import publish_to_devto, trigger_cms_webhook
from src.ui.renderer import render_markdown_with_local_images
from src.paths import IMAGES_DIR


def render_plan_tab(out: Dict[str, Any]):
    st.subheader("🧩 Article Execution Plan")
    plan_obj = out.get("plan")
    if not plan_obj:
        st.info("No plan details found in the output.")
        return

    if hasattr(plan_obj, "model_dump"):
        plan_dict = plan_obj.model_dump()
    elif isinstance(plan_obj, dict):
        plan_dict = plan_obj
    else:
        plan_dict = json.loads(json.dumps(plan_obj, default=str))

    st.markdown(f"### **Title:** {plan_dict.get('blog_title', 'Untitled')}")
    cols = st.columns(3)
    cols[0].metric("Target Audience", str(plan_dict.get("audience", "General")))
    cols[1].metric("Content Tone", str(plan_dict.get("tone", "Technical")))
    cols[2].metric("Article Format", str(plan_dict.get("blog_kind", "explainer")).upper())

    tasks = plan_dict.get("tasks", [])
    if tasks:
        st.markdown("#### **Section Breakdown & Specifications**")
        df = pd.DataFrame(
            [
                {
                    "ID": t.get("id"),
                    "Section Title": t.get("title"),
                    "Target Words": t.get("target_words"),
                    "Needs Research": "Yes" if t.get("requires_research") else "No",
                    "Citations": "Yes" if t.get("requires_citations") else "No",
                    "Code Snippet": "Yes" if t.get("requires_code") else "No",
                    "Tags": ", ".join(t.get("tags") or []),
                }
                for t in tasks
            ]
        ).sort_values("ID")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🔍 View Raw Task Specifications"):
            st.json(tasks)


def render_evidence_tab(out: Dict[str, Any]):
    st.subheader("🔎 Research Evidence & Citation Sources")
    evidence = out.get("evidence") or []
    if not evidence:
        st.info("No web evidence gathered for this run.")
        return

    rows = []
    for e in evidence:
        if hasattr(e, "model_dump"):
            e = e.model_dump()
        rows.append(
            {
                "Title": e.get("title"),
                "Published Date": e.get("published_at") or "Unknown",
                "Source Domain": e.get("source") or "Web",
                "URL": e.get("url"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📄 Detailed Evidence Snippets"):
        for item in evidence:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            st.markdown(f"**[{item.get('title')}]({item.get('url')})**")
            if item.get("snippet"):
                st.caption(f"> {item.get('snippet')}")
            st.divider()


def render_preview_tab(out: Dict[str, Any]):
    final_md = out.get("final") or ""
    if not final_md:
        st.warning("No generated markdown available to preview.")
        return

    plan_obj = out.get("plan")
    if hasattr(plan_obj, "blog_title"):
        blog_title = plan_obj.blog_title
    elif isinstance(plan_obj, dict):
        blog_title = plan_obj.get("blog_title", "article")
    else:
        blog_title = extract_title_from_md(final_md, "article")

    words_count = len(final_md.split())
    read_mins = max(1, round(words_count / 220))
    sources_count = len(out.get("evidence") or [])

    st.markdown(
        f"""
        <div class="article-meta">
            <strong>By AI Blog Writing Agent</strong> · {read_mins} min read · {sources_count} sources · {words_count:,} words
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_markdown_with_local_images(final_md)

    st.divider()
    st.markdown("### 📥 Multi-Format Export Engine")
    
    md_filename = f"{safe_slug(blog_title)}.md"
    md_frontmatter = export_to_markdown_with_frontmatter(
        final_md,
        out.get("seo_plan"),
        out.get("social_posts"),
    )
    html_content = export_to_html(final_md)
    json_report = export_to_json(out)
    bundle = bundle_zip(final_md, md_filename, IMAGES_DIR)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            "⬇️ Markdown (.md)",
            data=md_frontmatter.encode("utf-8"),
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True,
            help="Download clean Markdown with Hugo/Jekyll YAML frontmatter and social posts.",
        )
    with col2:
        st.download_button(
            "🌐 HTML (.html)",
            data=html_content.encode("utf-8"),
            file_name=f"{safe_slug(blog_title)}.html",
            mime="text/html",
            use_container_width=True,
            help="Download standalone styled HTML document.",
        )
    with col3:
        st.download_button(
            "📊 Full Report (.json)",
            data=json_report.encode("utf-8"),
            file_name=f"{safe_slug(blog_title)}_report.json",
            mime="application/json",
            use_container_width=True,
            help="Download complete JSON payload (Article, SEO, Fact Check, Social Posts).",
        )
    with col4:
        st.download_button(
            "📦 Bundle (.zip)",
            data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            help="Download markdown and all generated diagrams/images as a ZIP package.",
        )


def render_images_tab(out: Dict[str, Any]):
    st.subheader("🖼️ Diagrams & Visual Assets")
    specs = out.get("image_specs") or []
    images_dir = IMAGES_DIR

    if not specs and not images_dir.exists():
        st.info("No technical diagrams or images were generated for this post.")
        return

    if specs:
        st.markdown("#### **Image Specifications & Prompts**")
        st.json(specs)

    if images_dir.exists():
        files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.svg')
        ]
        if not files:
            st.warning("Image directory exists but contains no image files.")
        else:
            st.markdown("#### **Generated Visual Artifacts**")
            cols = st.columns(min(len(files), 3))
            for i, p in enumerate(sorted(files, key=lambda x: str(x))):
                rel_name = str(p.relative_to(images_dir))
                with cols[i % len(cols)]:
                    st.image(str(p), caption=rel_name, use_container_width=True)

            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download All Images (.zip)",
                    data=z,
                    file_name="article_images.zip",
                    mime="application/zip",
                )


def render_critic_tab(out: Dict[str, Any]):
    st.subheader("🧐 Autonomous Critic & Quality Audit")
    eval_obj = out.get("critic_evaluation")
    if not eval_obj:
        st.info("No critic evaluation details available for this run.")
        return

    if hasattr(eval_obj, "model_dump"):
        eval_dict = eval_obj.model_dump()
    elif isinstance(eval_obj, dict):
        eval_dict = eval_obj
    else:
        eval_dict = json.loads(json.dumps(eval_obj, default=str))

    score = eval_dict.get("score", 0.0)
    passed = eval_dict.get("passed", False)
    summary = eval_dict.get("summary", "")
    issues = eval_dict.get("issues", [])
    revisions_done = out.get("critic_retry_count", 0)

    cols = st.columns(3)
    cols[0].metric("OVERALL QUALITY SCORE", f"{score:.1f} / 10")
    cols[1].metric("AUDIT STATUS", "✅ APPROVED" if passed else "⚠️ REVISION NEEDED")
    cols[2].metric("REVISIONS PERFORMED", f"{revisions_done} revision(s)")

    if summary:
        st.markdown("#### **Evaluation Summary**")
        st.info(summary)

    if issues:
        st.markdown("#### **Identified Quality Issues & Recommendations**")
        rows = []
        for i in issues:
            if hasattr(i, "model_dump"):
                i = i.model_dump()
            rows.append({
                "Category": str(i.get("category", "")).capitalize(),
                "Severity": str(i.get("severity", "")).upper(),
                "Description": i.get("description"),
                "Actionable Suggestion": i.get("suggestion"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📄 View Detailed Critic Audit JSON"):
            st.json(eval_dict)


def render_logs_tab(logs: List[str]):
    st.subheader("🧾 Technical Run Logs")
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if logs:
        st.session_state["logs"].extend(logs)

    all_logs = "\n\n".join(st.session_state["logs"][-100:])
    st.text_area("Live Execution Stream Log", value=all_logs, height=500)


def render_fact_check_tab(out: Dict[str, Any]):
    st.subheader("🛡️ Fact & Citation Grounding Audit")
    fc_obj = out.get("fact_check_report")
    if not fc_obj:
        st.info("No fact check audit details available for this run.")
        return

    if hasattr(fc_obj, "model_dump"):
        fc_dict = fc_obj.model_dump()
    elif isinstance(fc_obj, dict):
        fc_dict = fc_obj
    else:
        fc_dict = json.loads(json.dumps(fc_obj, default=str))

    score = fc_dict.get("score", 0.0)
    claims = fc_dict.get("claims", [])
    total = fc_dict.get("total_claims_checked", len(claims))
    supported_count = len([
        c for c in claims
        if (c.get("classification") if isinstance(c, dict) else getattr(c, "classification", "")) in ("SUPPORTED", "PARTIALLY_SUPPORTED") or (c.get("verdict") if isinstance(c, dict) else "") == "verified"
    ])

    cols = st.columns(3)
    cols[0].metric("FACT GROUNDING SCORE", f"{score:.1f} / 10")
    cols[1].metric("CLAIMS SUPPORTED", f"{supported_count} / {total}")
    cols[2].metric("GROUNDING RATE", f"{(supported_count/max(1,total))*100:.0f}%")

    if claims:
        st.markdown("#### **Atomic Claim Verification Matrix**")
        rows = []
        for c in claims:
            if hasattr(c, "model_dump"):
                c = c.model_dump()
            classification = str(c.get("classification") or c.get("verdict", "")).upper()
            if classification in ("SUPPORTED", "VERIFIED"):
                icon = "✅ SUPPORTED"
            elif classification == "PARTIALLY_SUPPORTED":
                icon = "⚡ PARTIALLY SUPPORTED"
            elif classification == "CONTRADICTED":
                icon = "❌ CONTRADICTED"
            else:
                icon = "⚠️ UNSUPPORTED"

            sources = c.get("supporting_sources") or c.get("contradicting_sources") or []
            src_str = ", ".join(sources) if sources else (c.get("source_url") or "Evidence Pack")

            rows.append({
                "Status": icon,
                "Factual Claim": c.get("claim"),
                "Evidence Source": src_str,
                "Explanation": c.get("explanation"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_seo_tab(out: Dict[str, Any]):
    st.subheader("🎯 SEO & Search Optimization Package")
    seo_obj = out.get("seo_plan")
    if not seo_obj:
        st.info("No SEO optimization details available for this run.")
        return

    if hasattr(seo_obj, "model_dump"):
        seo_dict = seo_obj.model_dump()
    elif isinstance(seo_obj, dict):
        seo_dict = seo_obj
    else:
        seo_dict = json.loads(json.dumps(seo_obj, default=str))

    score = seo_dict.get("seo_score", 0.0)
    cols = st.columns(3)
    cols[0].metric("SEO SCORE", f"{score:.1f} / 10")
    cols[1].metric("READABILITY LEVEL", str(seo_dict.get("target_readability_level", "Intermediate")))
    cols[2].metric("SUGGESTED SLUG", f"/{seo_dict.get('suggested_slug', 'article')}")

    st.markdown("#### **Search Engine Snippet Preview**")
    meta_title = seo_dict.get("meta_title", "")
    meta_desc = seo_dict.get("meta_description", "")
    slug = seo_dict.get("suggested_slug", "")

    st.markdown(
        f"""
        <div style="background: #1E293B; border-radius: 8px; padding: 1rem; border: 1px solid #334155; margin-bottom: 1rem;">
            <div style="color: #60A5FA; font-size: 0.8rem; margin-bottom: 2px;">https://yourblog.com › posts › {slug}</div>
            <div style="color: #38BDF8; font-size: 1.15rem; font-weight: 600; margin-bottom: 4px;">{html.escape(meta_title)}</div>
            <div style="color: #94A3B8; font-size: 0.9rem;">{html.escape(meta_desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown(f"**Primary Keyword:** `{seo_dict.get('primary_keyword', '')}`")
    with col_k2:
        sec_kw = ", ".join(seo_dict.get("secondary_keywords", []))
        st.markdown(f"**Secondary Keywords:** `{sec_kw}`")

    faqs = seo_dict.get("faq_items", [])
    if faqs:
        st.markdown("#### **Frequently Asked Questions (FAQ)**")
        for item in faqs:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            with st.expander(f"❓ {item.get('question')}"):
                st.write(item.get("answer"))


def render_social_posts_tab(out: Dict[str, Any]):
    st.subheader("📢 Social Media Syndication & Derivative Content")
    social_obj = out.get("social_posts")
    if not social_obj:
        st.info("No social syndication posts available for this run.")
        return

    if hasattr(social_obj, "model_dump"):
        social_dict = social_obj.model_dump()
    elif isinstance(social_obj, dict):
        social_dict = social_obj
    else:
        social_dict = {}

    st.markdown("#### **🐦 Twitter / X Thread**")
    tweets = social_dict.get("twitter_thread", [])
    if tweets:
        for idx, tweet in enumerate(tweets, 1):
            st.text_area(f"Tweet {idx}/{len(tweets)} ({len(tweet)} chars)", value=tweet, height=85, key=f"tw_{idx}")
    else:
        st.caption("No Twitter thread available.")

    st.divider()
    st.markdown("#### **💼 LinkedIn Post**")
    linkedin = social_dict.get("linkedin_post", "")
    if linkedin:
        st.text_area("LinkedIn Post Content", value=linkedin, height=220, key="linkedin_text")
    else:
        st.caption("No LinkedIn post available.")

    st.divider()
    st.markdown("#### **📧 Email Newsletter Summary**")
    newsletter = social_dict.get("newsletter_summary", "")
    if newsletter:
        st.text_area("Newsletter Digest", value=newsletter, height=180, key="newsletter_text")
    else:
        st.caption("No newsletter summary available.")


def render_publishing_tab(out: Dict[str, Any]):
    st.subheader("🚀 Webhook & CMS Direct Publishing")
    st.caption("Publish your finished article directly to Dev.to or send a JSON payload to a custom CMS Webhook endpoint.")

    final_md = out.get("final") or ""
    plan_obj = out.get("plan")
    title = extract_title_from_md(final_md, "Untitled Article")
    if plan_obj:
        title = getattr(plan_obj, "blog_title", title) if hasattr(plan_obj, "blog_title") else plan_obj.get("blog_title", title)

    tags = []
    seo_plan = out.get("seo_plan")
    if seo_plan:
        kw = getattr(seo_plan, "primary_keyword", None) or (seo_plan.get("primary_keyword") if isinstance(seo_plan, dict) else None)
        if kw:
            tags.append(str(kw))

    with st.expander("📝 Dev.to Direct Publishing Options", expanded=True):
        devto_key = st.text_input("Dev.to API Key", type="password", key="devto_api_key_input", help="Get your API key at dev.to/settings/extensions")
        publish_immediately = st.checkbox("Publish immediately (Unchecked = Save as Draft)", value=False, key="devto_publish_mode")

        if st.button("🚀 Publish to Dev.to", key="devto_pub_btn", use_container_width=True):
            if not devto_key:
                st.error("Please enter a valid Dev.to API Key.")
            else:
                with st.spinner("Publishing to Dev.to..."):
                    res = publish_to_devto(
                        api_key=devto_key,
                        title=title,
                        markdown_content=final_md,
                        tags=tags,
                        published=publish_immediately,
                    )
                    if res.get("status") == "success":
                        st.success(f"Successfully published! [View Article on Dev.to]({res.get('url')})")
                    else:
                        st.error(f"Publishing failed: {res.get('message')}")

    with st.expander("📡 Custom CMS Webhook Options", expanded=True):
        webhook_url = st.text_input("Webhook Endpoint URL", key="webhook_url_input", placeholder="https://api.strapi.io/webhooks or https://hooks.zapier.com/...")

        if st.button("📡 Trigger CMS Webhook", key="webhook_pub_btn", use_container_width=True):
            if not webhook_url:
                st.error("Please enter a valid Webhook URL.")
            else:
                with st.spinner("Sending payload to Webhook..."):
                    payload = {
                        "title": title,
                        "article_markdown": final_md,
                        "seo_metadata": out.get("seo_plan").model_dump() if hasattr(out.get("seo_plan"), "model_dump") else out.get("seo_plan"),
                        "social_posts": out.get("social_posts").model_dump() if hasattr(out.get("social_posts"), "model_dump") else out.get("social_posts"),
                    }
                    res = trigger_cms_webhook(webhook_url, payload)
                    if res.get("status") == "success":
                        st.success(f"Webhook triggered successfully! Status Code: {res.get('status_code')}")
                    else:
                        st.error(f"Webhook trigger failed: {res.get('message')}")


def render_article_workspace(out: Dict[str, Any], is_history: bool = False, blog_name: str = ""):
    """
    Renders workspace for an article.
    If viewing a past history blog, shows ONLY the Article preview and metrics banner.
    If viewing a newly generated article, shows full execution tabs.
    """
    if is_history:
        st.info(f"📂 **Viewing Library Article:** `{blog_name}`")
        render_article_metrics(out)
        st.divider()
        render_preview_tab(out)
    else:
        render_article_metrics(out)
        tab_article, tab_social, tab_publish, tab_seo, tab_fc, tab_critic, tab_research, tab_plan, tab_visuals, tab_logs = st.tabs(
            ["📝 Article", "📢 Social", "🚀 Publish", "🎯 SEO", "🛡️ Fact Check", "🧐 Quality Audit", "🔎 Research", "🧩 Plan", "🖼️ Visuals", "🧾 Run Details"]
        )
        with tab_article:
            render_preview_tab(out)
        with tab_social:
            render_social_posts_tab(out)
        with tab_publish:
            render_publishing_tab(out)
        with tab_seo:
            render_seo_tab(out)
        with tab_fc:
            render_fact_check_tab(out)
        with tab_critic:
            render_critic_tab(out)
        with tab_research:
            render_evidence_tab(out)
        with tab_plan:
            render_plan_tab(out)
        with tab_visuals:
            render_images_tab(out)
        with tab_logs:
            render_logs_tab([])



def cb_back_to_archive():
    st.session_state["selected_past_article"] = None


def cb_open_past_article(filename: str):
    st.session_state["selected_past_article"] = filename
    st.session_state["app_mode"] = "📜 Past Articles"


def render_library_workspace():
    """
    Renders a dedicated page to view, read, and browse old/past articles.
    Supports dedicated full-screen article reading view with a back navigation button.
    """
    selected_file = st.session_state.get("selected_past_article")

    if selected_file:
        past_files = list_past_blogs()
        target_path = next((p for p in past_files if p.name == selected_file), None)

        if target_path:
            col_back, col_space = st.columns([2.5, 5])
            with col_back:
                st.button(
                    "⬅️ Back to All Past Articles",
                    use_container_width=True,
                    type="secondary",
                    key="back_to_archive_btn",
                    on_click=cb_back_to_archive,
                )

            try:
                md_text = read_md_file(target_path)
                title = extract_title_from_md(md_text, target_path.stem)
            except Exception:
                md_text = ""
                title = target_path.stem

            st.info(f"📜 **Viewing Past Article:** `{title}` (`{target_path.name}`)")

            out = {
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "final": md_text,
            }
            render_article_metrics(out)
            st.divider()
            render_preview_tab(out)
            return

    st.subheader("📜 Past Articles Archive")
    st.caption("Browse, read, and export your previously generated technical articles.")

    past_files = list_past_blogs()
    if not past_files:
        st.info("No saved articles found in the outputs archive.")
        return

    cols = st.columns(2)
    for idx, p in enumerate(past_files):
        try:
            md_text = read_md_file(p)
            title = extract_title_from_md(md_text, p.stem)
            words = len(md_text.split())
            read_time = max(1, round(words / 220))
            mtime = p.stat().st_mtime
            import datetime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime("%d %b %Y")
        except Exception:
            title = p.stem
            words = 0
            read_time = 1
            date_str = "Unknown"

        with cols[idx % 2]:
            with st.container():
                st.markdown(
                    f"""
                    <div class="library-card">
                        <h4 style="margin-bottom: 0.5rem; color: #F8FAFC;">{html.escape(title)}</h4>
                        <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 1rem;">
                            📅 {html.escape(str(date_str))} · ⏱️ {read_time} min read ({words:,} words)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "📖 Read Full Article",
                    key=f"lib_btn_{p.name}",
                    use_container_width=True,
                    on_click=cb_open_past_article,
                    args=(p.name,),
                )
