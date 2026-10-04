"""Exportador OOXML pequeno: texto editável, imagens e notas do apresentador."""

from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
GROUP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
         '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
         '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


def rels(items):
    return XML + f'<Relationships xmlns="{PKG}">' + "".join(
        f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{escape(target)}"/>'
        for rid, kind, target in items) + '</Relationships>'


def transform(element):
    x, y, w, h = [round(element[key] * 914400) for key in ["x", "y", "w", "h"]]
    return f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'


def paragraphs(text, size, color, bold=False):
    props = (f'<a:rPr lang="pt-BR" sz="{round(size * 100)}" b="{int(bold)}">'
             f'<a:solidFill><a:srgbClr val="{color.lstrip(chr(35))}"/></a:solidFill>'
             '<a:latin typeface="DejaVu Sans"/></a:rPr>')
    return "".join('<a:p><a:pPr><a:lnSpc><a:spcPct val="110000"/></a:lnSpc></a:pPr>'
                   f'<a:r>{props}<a:t xml:space="preserve">{escape(line)}</a:t></a:r>'
                   f'<a:endParaRPr lang="pt-BR" sz="{round(size * 100)}"/></a:p>'
                   for line in text.splitlines())


def shape(element, index):
    name = f"Elemento {index}"
    if element["type"] == "image":
        return (f'<p:pic><p:nvPicPr><p:cNvPr id="{index}" name="{name}"/>'
                '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
                '<p:blipFill><a:blip r:embed="rIdImage"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
                f'<p:spPr>{transform(element)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                '</p:spPr></p:pic>')
    fill = element.get("fill", "FFFFFF").lstrip("#")
    sp = (f'<p:sp><p:nvSpPr><p:cNvPr id="{index}" name="{name}"/>'
          f'<p:cNvSpPr txBox="{int(element["type"] == "text")}"/><p:nvPr/></p:nvSpPr>'
          f'<p:spPr>{transform(element)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>')
    if element["type"] == "rect":
        return sp + f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
    return (sp + '<a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody>'
            '<a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"/>'
            '<a:lstStyle/>' + paragraphs(element["text"], element["size"], element["color"],
                                       element.get("bold", False)) + '</p:txBody></p:sp>')


def write_pptx(path, slides, theme_source):
    """Salve slides nativos usando o tema do template existente apenas como base."""
    with ZipFile(theme_source) as template:
        theme = template.read("ppt/theme/theme1.xml")
    overrides = [('presentation', 'presentation.main'), ('slideMasters/slideMaster1', 'slideMaster'),
                 ('slideLayouts/slideLayout1', 'slideLayout'), ('notesMasters/notesMaster1', 'notesMaster')]
    for i in range(1, len(slides) + 1):
        overrides.extend([(f"slides/slide{i}", "slide"), (f"notesSlides/notesSlide{i}", "notesSlide")])
    content_types = (XML + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Default Extension="png" ContentType="image/png"/>'
                     '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
                     + "".join(f'<Override PartName="/ppt/{part}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml"/>'
                               for part, kind in overrides) + '</Types>')
    ns = f'xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"'
    clr = ('<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
           'accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" '
           'hlink="hlink" tx1="dk1" tx2="dk2"/>')
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels([("rId1", "officeDocument", "ppt/presentation.xml")]))
        archive.writestr("ppt/theme/theme1.xml", theme)
        archive.writestr("ppt/presentation.xml", XML + f'<p:presentation {ns}>'
                         '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rIdMaster"/></p:sldMasterIdLst>'
                         '<p:notesMasterIdLst><p:notesMasterId r:id="rIdNotesMaster"/></p:notesMasterIdLst>'
                         '<p:sldIdLst>' + "".join(f'<p:sldId id="{255+i}" r:id="rIdSlide{i}"/>' for i in range(1, len(slides)+1))
                         + '</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>'
                         '<p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>')
        archive.writestr("ppt/_rels/presentation.xml.rels", rels(
            [("rIdMaster", "slideMaster", "slideMasters/slideMaster1.xml"),
             ("rIdNotesMaster", "notesMaster", "notesMasters/notesMaster1.xml")]
            + [(f"rIdSlide{i}", "slide", f"slides/slide{i}.xml") for i in range(1, len(slides)+1)]))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", XML + f'<p:sldMaster {ns}>'
                         f'<p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{clr}'
                         '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
                         '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels(
            [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "theme", "../theme/theme1.xml")]))
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", XML + f'<p:sldLayout {ns} type="blank" preserve="1">'
                         f'<p:cSld name="Blank"><p:spTree>{GROUP}</p:spTree></p:cSld>'
                         '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels(
            [("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")]))
        archive.writestr("ppt/notesMasters/notesMaster1.xml", XML + f'<p:notesMaster {ns}>'
                         f'<p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{clr}<p:notesStyle/></p:notesMaster>')
        archive.writestr("ppt/notesMasters/_rels/notesMaster1.xml.rels", rels(
            [("rId1", "theme", "../theme/theme1.xml")]))
        for i, slide in enumerate(slides, 1):
            archive.writestr(f"ppt/slides/slide{i}.xml", XML + f'<p:sld {ns}>'
                             '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
                             '<a:effectLst/></p:bgPr></p:bg><p:spTree>' + GROUP
                             + "".join(shape(e, j) for j, e in enumerate(slide["elements"], 2))
                             + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
            relationships = [("rIdLayout", "slideLayout", "../slideLayouts/slideLayout1.xml"),
                             ("rIdNotes", "notesSlide", f"../notesSlides/notesSlide{i}.xml")]
            images = [e for e in slide["elements"] if e["type"] == "image"]
            if images:
                media = f"figure{i}.png"
                archive.writestr(f"ppt/media/{media}", Path(images[0]["path"]).read_bytes())
                relationships.append(("rIdImage", "image", f"../media/{media}"))
            archive.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels(relationships))
            note_element = {"type": "text", "text": slide["notes"], "x": .5, "y": 1,
                            "w": 6.5, "h": 7, "size": 14, "color": "000000"}
            note_shape = shape(note_element, 2).replace('<p:nvPr/>', '<p:nvPr><p:ph type="body" idx="1"/></p:nvPr>')
            archive.writestr(f"ppt/notesSlides/notesSlide{i}.xml", XML + f'<p:notes {ns}>'
                             f'<p:cSld><p:spTree>{GROUP}{note_shape}</p:spTree></p:cSld>'
                             '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')
            archive.writestr(f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels", rels(
                [("rId1", "notesMaster", "../notesMasters/notesMaster1.xml"),
                 ("rId2", "slide", f"../slides/slide{i}.xml")]))
