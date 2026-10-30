from datetime import date, time
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, Time, Text
from sqlalchemy.orm import Mapped, relationship

from app.db.database import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = Column(String, unique=True, nullable=False, index=True)
    duration: Mapped[float] = Column(Float, nullable=False)
    fps: Mapped[float] = Column(Float, nullable=False)
    upload_date: Mapped[date] = Column(Date, nullable=False)
    upload_time: Mapped[time] = Column(Time, nullable=False)
    file_hash: Mapped[str | None] = Column(String, nullable=True, index=True)
    status: Mapped[str] = Column(String, nullable=False, default="uploading")
    progress: Mapped[float] = Column(Float, nullable=False, default=0.0)

    frames: Mapped[list["Frame"]] = relationship(
        "Frame",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="joined",
    )

    events: Mapped[list["Event"]] = relationship(
        "Event",
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="select",
    )


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    frame_name: Mapped[str] = Column(String, nullable=False)
    timestamp: Mapped[float] = Column(Float, nullable=False)

    video: Mapped[Video] = relationship("Video", back_populates="frames")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection",
        back_populates="frame",
        cascade="all, delete-orphan",
        lazy="joined",
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    label: Mapped[str] = Column(String, nullable=False, index=True)
    start: Mapped[float] = Column(Float, nullable=False)
    end: Mapped[float] = Column(Float, nullable=False)
    clip_path: Mapped[str | None] = Column(Text, nullable=True)
    max_conf: Mapped[float] = Column(Float, nullable=False, default=0.0)

    video: Mapped[Video] = relationship("Video", back_populates="events")
